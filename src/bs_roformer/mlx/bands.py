"""Frequency band-split and mask-estimation modules for the BS-RoFormer trunk.

`BandSplit` projects each frequency band's STFT bins into the model's feature
dimension; `MaskEstimator` runs the inverse, turning transformer output back into
a per-band complex mask via a per-band GLU-MLP (`MLP` below). Both carry an
opt-in "grouped" fast path (`MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT` /
`_GROUPED_MASK_ESTIMATOR`) that batches bands sharing the same input width into
one matmul via `ops.batched_group_linear`, with an optional weight-pack cache
(`_GROUPED_WEIGHT_CACHE`) keyed on parameter identity so the packed tensors are
only rebuilt when weights actually change. `BSRoformerBlock` is a thin container
only -- the actual per-block forward order (linear -> time -> freq transformer)
lives in `model.py`'s `_forward_transformers`, not here, so this file owns band
knowledge and none of the orchestration.

Reads: .ops (batched_group_linear, default, env_enabled)
"""

import itertools
from collections import defaultdict
from typing import Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402
import numpy as np

from .attention import L2Norm
from .ops import batched_group_linear, default, env_enabled


class BandSplitModule(nn.Module):
    """Single band processing module."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = L2Norm(dim_in)
        self.linear = nn.Linear(dim_in, dim_out)

    def __call__(self, x):
        return self.linear(self.norm(x))


class BandSplit(nn.Module):
    """
    Band-split module that splits frequency bins into bands and projects to feature dimension.
    """

    def __init__(self, dim, dim_inputs: Tuple[int, ...], use_grouped: bool = False):  # noqa: UP006
        super().__init__()
        self.dim_inputs = dim_inputs
        self.num_bands = len(dim_inputs)
        self.split_points = np.cumsum(self.dim_inputs)[:-1].tolist()
        self.use_grouped = bool(use_grouped)
        self._band_module_names: list[str] = []
        grouped: dict[int, list[int]] = defaultdict(list)

        # Store as individual attributes for proper MLX registration
        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_features_{i}"
            setattr(self, module_name, BandSplitModule(dim_in, dim))
            self._band_module_names.append(module_name)
            grouped[int(dim_in)].append(i)
        self._grouped_band_indices = tuple((dim_in, tuple(indices)) for dim_in, indices in grouped.items())
        self.use_grouped_weight_cache = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE",
            default_value=False,
        )
        self._grouped_pack_cache: dict[tuple[int, ...], dict[str, object]] = {}

    def _get_grouped_pack(self, band_indices: tuple[int, ...]) -> dict[str, object]:
        modules = [getattr(self, self._band_module_names[idx]) for idx in band_indices]
        signature_items: list[int] = []
        for module in modules:
            signature_items.append(id(module.norm.weight))
            signature_items.append(id(module.linear.weight))
            bias = getattr(module.linear, "bias", None)
            signature_items.append(id(bias) if bias is not None else 0)
        signature = tuple(signature_items)

        if self.use_grouped_weight_cache:
            cached = self._grouped_pack_cache.get(band_indices)
            if cached is not None and cached.get("signature") == signature:
                return cached

        norm_weights = mx.stack([module.norm.weight for module in modules], axis=0)
        linear_weights = mx.stack([module.linear.weight for module in modules], axis=0)
        biases = [getattr(module.linear, "bias", None) for module in modules]
        has_bias = all(bias is not None for bias in biases)
        linear_bias = mx.stack(biases, axis=0) if has_bias else None
        packed = {
            "signature": signature,
            "eps": float(modules[0].norm.eps),
            "scale": float(modules[0].norm.scale),
            "norm_weights": norm_weights,
            "linear_weights": linear_weights,
            "linear_bias": linear_bias,
        }
        if self.use_grouped_weight_cache:
            self._grouped_pack_cache[band_indices] = packed
        return packed

    def _forward_grouped(self, splits: list[mx.array]) -> mx.array:
        outs: list[mx.array | None] = [None] * self.num_bands

        for _, band_indices in self._grouped_band_indices:
            if len(band_indices) <= 1:
                band_idx = int(band_indices[0])
                to_feature = getattr(self, self._band_module_names[band_idx])
                outs[band_idx] = to_feature(splits[band_idx])
                continue

            grouped_input = mx.stack([splits[idx] for idx in band_indices], axis=2)  # (B, T, G, D)
            packed = self._get_grouped_pack(band_indices)

            # Grouped L2 norm equivalent to L2Norm module math.
            eps = float(packed["eps"])
            scale = float(packed["scale"])
            norm = mx.sqrt(mx.sum(grouped_input * grouped_input, axis=-1, keepdims=True))
            denom = mx.maximum(norm, eps)
            normalized = (grouped_input / denom) * scale
            norm_weights = packed["norm_weights"]
            normalized = normalized * norm_weights[None, None, :, :]
            linear_weights = packed["linear_weights"]
            linear_bias = packed["linear_bias"]
            grouped_out = batched_group_linear(normalized, linear_weights, linear_bias)
            for local_idx, band_idx in enumerate(band_indices):
                outs[int(band_idx)] = grouped_out[:, :, local_idx, :]

        return mx.stack([out for out in outs if out is not None], axis=-2)

    def __call__(self, x):
        # Split input by frequency bands
        splits = mx.split(x, self.split_points, axis=-1)
        if self.use_grouped:
            return self._forward_grouped(splits)

        outs = []
        for i, split_input in enumerate(splits):
            to_feature = getattr(self, self._band_module_names[i])
            split_output = to_feature(split_input)
            outs.append(split_output)

        return mx.stack(outs, axis=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1):
    """Helper to create MLP with MLX Sequential (stores layers in self.layers list)."""
    dim_hidden = default(dim_hidden, dim_in)

    layers = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(itertools.pairwise(dims)):
        is_last = ind == (len(dims) - 2)

        layers.append(nn.Linear(layer_dim_in, layer_dim_out))

        if not is_last:
            layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    """
    Mask estimator that generates frequency masks for each stem.
    """

    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4, use_grouped: bool = False):  # noqa: UP006
        super().__init__()
        self.dim_inputs = dim_inputs
        self.num_bands = len(dim_inputs)
        self.use_grouped = bool(use_grouped)
        dim_hidden = dim * mlp_expansion_factor
        self._mlp_module_names: list[str] = []
        grouped: dict[int, list[int]] = defaultdict(list)

        # Store as individual attributes for proper MLX registration
        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_freqs_{i}"
            setattr(self, module_name, MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth))
            self._mlp_module_names.append(module_name)
            grouped[int(dim_in)].append(i)
        self._grouped_band_indices = tuple((dim_in, tuple(indices)) for dim_in, indices in grouped.items())
        self.use_grouped_weight_cache = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE",
            default_value=False,
        )
        self._grouped_mlp_pack_cache: dict[tuple[int, ...], dict[str, object] | None] = {}

    def _get_grouped_mlp_pack(self, band_indices: tuple[int, ...]) -> dict[str, object] | None:
        mlps = [getattr(self, self._mlp_module_names[idx]) for idx in band_indices]
        if not mlps:
            return None

        layer_lists = [getattr(mlp, "layers", None) for mlp in mlps]
        if any(layers is None for layers in layer_lists):
            return None
        depth = len(layer_lists[0])
        if any(len(layers) != depth for layers in layer_lists):
            return None

        metadata: list[tuple[str, list[mx.array] | None, list[mx.array | None] | None, bool]] = []
        signature_items: list[int] = []
        for layer_idx in range(depth):
            proto = layer_lists[0][layer_idx]
            proto_name = proto.__class__.__name__.lower()
            if hasattr(proto, "weight"):
                weights = []
                biases = []
                has_bias = True
                for layers in layer_lists:
                    layer = layers[layer_idx]
                    weight = getattr(layer, "weight", None)
                    if weight is None:
                        return None
                    weights.append(weight)
                    signature_items.append(id(weight))
                    bias = getattr(layer, "bias", None)
                    if bias is None:
                        has_bias = False
                    biases.append(bias)
                    signature_items.append(id(bias) if bias is not None else 0)
                metadata.append(("linear", weights, biases, has_bias))
            elif proto_name == "tanh":
                metadata.append(("tanh", None, None, False))
            else:
                return None

        signature = tuple(signature_items)
        if self.use_grouped_weight_cache:
            cached = self._grouped_mlp_pack_cache.get(band_indices)
            if cached is not None and cached.get("signature") == signature:
                return cached

        ops: list[dict[str, object]] = []
        for kind, weights, biases, has_bias in metadata:
            if kind == "tanh":
                ops.append({"kind": "tanh"})
                continue
            assert weights is not None
            linear_weights = mx.stack(weights, axis=0)
            linear_bias = mx.stack(biases, axis=0) if has_bias and biases is not None else None
            ops.append({"kind": "linear", "weights": linear_weights, "bias": linear_bias})

        packed = {"signature": signature, "ops": ops}
        if self.use_grouped_weight_cache:
            self._grouped_mlp_pack_cache[band_indices] = packed
        return packed

    def _run_grouped_mlp(self, grouped_input: mx.array, band_indices: tuple[int, ...]) -> mx.array | None:
        packed = self._get_grouped_mlp_pack(band_indices)
        if packed is None:
            return None
        x = grouped_input
        for op in packed["ops"]:
            if op["kind"] == "tanh":
                x = mx.tanh(x)
            else:
                x = batched_group_linear(x, op["weights"], op["bias"])
        return x

    def __call__(self, x):
        # Unbind bands
        x_bands = [x[..., i, :] for i in range(x.shape[-2])]

        if self.use_grouped:
            outs_by_band: list[mx.array | None] = [None] * self.num_bands
            for _, band_indices in self._grouped_band_indices:
                if len(band_indices) <= 1:
                    band_idx = int(band_indices[0])
                    mlp = getattr(self, self._mlp_module_names[band_idx])
                    freq_out_before_glu = mlp(x_bands[band_idx])
                    freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
                    outs_by_band[band_idx] = freq_out[0] * mx.sigmoid(freq_out[1])
                    continue

                grouped_input = mx.stack([x_bands[idx] for idx in band_indices], axis=2)  # (B, T, G, D)
                grouped_out = self._run_grouped_mlp(grouped_input, band_indices)
                if grouped_out is None:
                    # Conservative fallback for unsupported mixed module structures.
                    for band_idx in band_indices:
                        mlp = getattr(self, self._mlp_module_names[int(band_idx)])
                        freq_out_before_glu = mlp(x_bands[int(band_idx)])
                        freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
                        outs_by_band[int(band_idx)] = freq_out[0] * mx.sigmoid(freq_out[1])
                    continue

                values, gates = mx.split(grouped_out, 2, axis=-1)
                grouped_masks = values * mx.sigmoid(gates)
                for local_idx, band_idx in enumerate(band_indices):
                    outs_by_band[int(band_idx)] = grouped_masks[:, :, local_idx, :]

            return mx.concatenate([out for out in outs_by_band if out is not None], axis=-1)

        outs = []
        for i, band_features in enumerate(x_bands):
            mlp = getattr(self, self._mlp_module_names[i])
            freq_out_before_glu = mlp(band_features)

            # Apply GLU (Gated Linear Unit)
            freq_out = mx.split(freq_out_before_glu, 2, axis=-1)
            freq_out = freq_out[0] * mx.sigmoid(freq_out[1])

            outs.append(freq_out)

        return mx.concatenate(outs, axis=-1)


class BSRoformerBlock(nn.Module):
    """
    Single BS-Roformer block containing time and frequency transformers.
    Optionally includes linear transformer.
    """
    def __init__(self, linear_transformer, time_transformer, freq_transformer):
        super().__init__()
        self.has_linear = linear_transformer is not None

        if self.has_linear:
            self.linear_transformer = linear_transformer
        self.time_transformer = time_transformer
        self.freq_transformer = freq_transformer

    def __call__(self, x):
        # Apply transformers as in the forward pass
        # This is just a container - actual logic is in BSRoformerMLX.__call__
        return x


# Default frequency band configuration (sums to 1025 for n_fft=2048)
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)
