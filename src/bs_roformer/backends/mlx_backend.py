"""MLX backend -- native Apple Silicon execution behind the same seam.

Builds the vendored MLX BS-RoFormer from the package's own config and its own
sha256-verified PyTorch checkpoint, so the package-owned checkpoint contract is
unchanged: no second catalog, no converted-weight cache the loader does not
control. Mixed precision is forced off because it was measured to buy no speed on
this path while degrading Torch-vs-MLX agreement by roughly 87x, and article 2
requires an accuracy-affecting optimization to be opt-in.

Two refusals are deliberate. A checkpoint whose registry `variation` names a
mask-estimator head with no MLX module present is rejected rather than run --
upstream's model swallows unknown constructor arguments and would otherwise build
a plain MaskEstimator and emit confident garbage. Support is *measured* from which
head modules exist (`supported_variations`), never declared, so a head that has not
been written cannot be advertised and then fail deep inside construction. And a
config whose chunk_size is not a multiple of its STFT hop is rejected, because the
chunked path's length bookkeeping silently assumes that alignment.

Reads: .base (ChunkingPlan, BackendUnavailable), ..mlx (BSRoformerMLX, conversion),
..mlx.heads (available_variants), ..utils (load_checkpoint_state), numpy
"""

from __future__ import annotations

import numpy as np

from .base import BackendUnavailable, ChunkingPlan


def supported_variations() -> frozenset:
    """Variations the MLX heads registry can actually build, plus empty sentinels.

    Measured from the heads package rather than restated here, so a head that has
    not been written yet cannot be advertised and then fail as an ImportError deep
    inside construction. Computed per call, not frozen at import, so this stays
    honest as heads land.
    """
    try:
        from ..mlx.heads import available_variants

        variants = available_variants()
    except ImportError:
        variants = ("mlp",)
    return frozenset({None, "", *variants})


class MLXBackend:
    """Runs BS-RoFormer natively on Apple Silicon through MLX."""

    name = "mlx"

    def __init__(self, model, config, device="mps"):
        self._model = model
        self._config = config
        self._device = device
        self._plan = ChunkingPlan.from_config(config)
        self._first_chunk_time = None

    # ------------------------------------------------------------- availability

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_spectro  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def _require(cls):
        if not cls.is_available():
            raise BackendUnavailable(
                "the MLX backend needs the optional extra: "
                "pip install 'bs-roformer-infer[mlx]' (Apple Silicon)"
            )

    # ------------------------------------------------------------- construction

    @classmethod
    def from_checkpoint(cls, *, config, checkpoint_path, variation=None, device="mps"):
        """Build an MLX model from this package's own checkpoint and config."""
        cls._require()
        cls.assert_supports_variation(variation)
        cls._reject_unaligned_chunking(config)

        from ..mlx import (
            BSRoformerMLX,
            convert_torch_to_mlx_weights,
            load_converted_weights,
        )
        from ..utils import load_checkpoint_state

        state = load_checkpoint_state(checkpoint_path, map_location="cpu")
        model = BSRoformerMLX(**cls._model_args(config, variation=variation))
        load_converted_weights(
            model, convert_torch_to_mlx_weights(state, variant=variation)
        )
        model.eval()
        return cls(model, config, device=device)

    @staticmethod
    def supports_variation(variation) -> bool:
        return variation in supported_variations()

    @classmethod
    def assert_supports_variation(cls, variation) -> None:
        if cls.supports_variation(variation):
            return
        raise BackendUnavailable(
            f"checkpoint variation {variation!r} has no MLX mask-estimator head; "
            f"use backend='torch' for this model. Supported here: "
            f"{sorted(v for v in supported_variations() if v)}"
        )

    @staticmethod
    def _reject_unaligned_chunking(config) -> None:
        plan = ChunkingPlan.from_config(config)
        hop = config.model.get("stft_hop_length", 512)
        if plan.chunk_size % hop:
            raise BackendUnavailable(
                f"chunk_size {plan.chunk_size} is not a multiple of stft_hop_length "
                f"{hop}; the chunked path's length bookkeeping assumes that alignment"
            )

    @staticmethod
    def _model_args(config, *, variation=None) -> dict:
        model_cfg = dict(config.model)
        args = {
            # The registry variation, not a config key: the Torch side injects it
            # the same way (utils.get_model_from_config), because the published
            # YAMLs do not carry it.
            "mask_estimator_variant": model_cfg.get("mask_estimator_variant")
            or variation
            or "mlp",
            "dim": model_cfg["dim"],
            "depth": model_cfg["depth"],
            "stereo": model_cfg.get("stereo", False),
            "num_stems": model_cfg.get("num_stems", 1),
            "time_transformer_depth": model_cfg.get("time_transformer_depth", 2),
            "freq_transformer_depth": model_cfg.get("freq_transformer_depth", 2),
            "linear_transformer_depth": model_cfg.get("linear_transformer_depth", 0),
            "dim_head": model_cfg.get("dim_head", 64),
            "heads": model_cfg.get("heads", 8),
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "mlp_expansion_factor": model_cfg.get("mlp_expansion_factor", 4),
            "mask_estimator_depth": model_cfg.get("mask_estimator_depth", 2),
            # Forwarded explicitly: upstream's own loader drops stft_normalized,
            # which is inert only while every config happens to leave it false.
            "stft_normalized": model_cfg.get("stft_normalized", False),
        }
        for key in ("stft_n_fft", "stft_hop_length", "stft_win_length"):
            if key in model_cfg:
                args[key] = model_cfg[key]
        # Absent from this package's own packaged config, where the model's own
        # default is the correct band split -- so only pass it when stated.
        if "freqs_per_bands" in model_cfg:
            args["freqs_per_bands"] = tuple(model_cfg["freqs_per_bands"])
        return args

    # ----------------------------------------------------------------- protocol

    @property
    def resolved_device(self) -> str:
        return self._device

    @property
    def model(self):
        """The resident MLX model, for callers composing the advanced path directly."""
        return self._model

    def release(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        """Chunked overlap-add, mirroring utils.demix_track's exact arithmetic."""
        import mlx.core as mx

        plan = self._plan
        training = self._config.training
        target = getattr(training, "target_instrument", None)
        stems = [target] if target is not None else list(training.instruments)

        audio = mx.array(np.ascontiguousarray(mix, dtype=np.float32))
        original_length = audio.shape[1]
        padded = original_length > 2 * plan.border and plan.border > 0
        if padded:
            audio = _reflect_pad(audio, plan.border, plan.border)

        total_length = audio.shape[1]
        window_template = _fade_window(mx, plan.chunk_size, plan.fade_size)
        shape = (len(stems), *audio.shape)
        result = mx.zeros(shape, dtype=mx.float32)
        counter = mx.zeros(shape, dtype=mx.float32)

        position = 0
        while position < total_length:
            part = audio[:, position:position + plan.chunk_size]
            length = part.shape[-1]
            if length < plan.chunk_size:
                missing = plan.chunk_size - length
                if length > plan.chunk_size // 2 + 1:
                    part = _reflect_pad(part, 0, missing)
                else:
                    part = mx.pad(part, [(0, 0), (0, missing)], constant_values=0.0)

            estimate = self._model(part[None])[0]

            window = _edge_corrected_window(
                mx, window_template, plan, position, total_length
            )
            span = slice(position, position + length)
            result[..., span] = result[..., span] + estimate[..., :length] * window[..., :length]
            counter[..., span] = counter[..., span] + window[..., :length]
            position += plan.step

        sources = np.array(result / counter, dtype=np.float32)
        np.nan_to_num(sources, copy=False, nan=0.0)
        if padded:
            sources = sources[..., plan.border:-plan.border]
        return dict(zip(stems, sources))


def _reflect_pad(array, left: int, right: int):
    """torch F.pad(mode='reflect') along the last axis; mx.pad has no reflect mode.

    Reflects without repeating the edge sample, which is torch's and numpy's
    convention -- an off-by-one here shifts every chunk boundary.
    """
    import mlx.core as mx

    if left:
        array = mx.concatenate([array[..., left:0:-1], array], axis=-1)
    if right:
        array = mx.concatenate([array, array[..., -2:-2 - right:-1]], axis=-1)
    return array


def _fade_window(mx, window_size: int, fade_size: int):
    window = np.ones(window_size, dtype=np.float32)
    window[:fade_size] *= np.linspace(0, 1, fade_size, dtype=np.float32)
    window[-fade_size:] *= np.linspace(1, 0, fade_size, dtype=np.float32)
    return mx.array(window)


def _edge_corrected_window(mx, template, plan: ChunkingPlan, position: int, total: int):
    """First and last chunks keep full amplitude at the outer edge, as torch does."""
    if position != 0 and position + plan.chunk_size < total:
        return template
    window = np.array(template, copy=True)
    if position == 0:
        window[:plan.fade_size] = 1.0
    else:
        window[-plan.fade_size:] = 1.0
    return mx.array(window)
