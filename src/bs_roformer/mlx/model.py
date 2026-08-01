"""BSRoformerMLX -- the BS-RoFormer trunk model, ported to Apple MLX.

Owned (not vendored) by this project: originally copied verbatim from upstream's
BS-RoFormer module, then reshaped once the project took ownership -- attention
and transformer primitives moved to `attention.py`, band-split and mask
estimation to `bands.py`, einops-lite tensor helpers to `ops.py`, and the
`exact_zero_safe_rfft` Metal-kernel workaround to `rfft_guard.py`. This file now
owns only the trunk: STFT -> band-split -> transformer stack -> mask estimation
-> masked iSTFT, i.e. `BSRoformerMLX.__init__` and `__call__` and their private
helpers. `mask_estimator_variant` is named explicitly (not swallowed through
`**kwargs`) so a checkpoint trained with a non-stock head cannot be silently
built with the wrong one -- selection is owned by `heads/__init__.py`.

Upstream provenance (kept for attribution; this file has since been modified --
see `CHANGELOG.md`):
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/separator/models/roformer/bs_roformer.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.

Reads: .attention (L2Norm, Transformer), .bands (BandSplit, BSRoformerBlock,
DEFAULT_FREQS_PER_BANDS), .ops (env_enabled, pack, rearrange, unpack),
.rfft_guard (exact_zero_safe_rfft), .heads (build_mask_estimator, lazily),
.variants (SiameseTransformer, ValueResidualTransformer),
mlx.core, mlx.nn, mlx_spectro
"""

import os
from typing import Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402
from mlx_spectro import get_transform_mlx

from .attention import L2Norm, Transformer
from .bands import DEFAULT_FREQS_PER_BANDS, BandSplit, BSRoformerBlock
from .ops import env_enabled, pack, rearrange, unpack
from .rfft_guard import exact_zero_safe_rfft


class BSRoformerMLX(nn.Module):
    """
    BS-Roformer (Band-Split RoFormer) for music source separation.

    MLX implementation leveraging optimized built-in components:
    - mlx.nn.RMSNorm for normalization
    - mlx.nn.RoPE for rotary position embeddings
    - mlx.core.fast.scaled_dot_product_attention for attention
    - mx.compile() for graph compilation

    This model achieves similar or better performance than PyTorch on Apple Silicon.
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # noqa: UP006
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=4,
        mask_estimator_depth=2,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        # NOT UPSTREAM: named explicitly so a variant checkpoint cannot be
        # swallowed by **kwargs and silently built as the stock head.
        mask_estimator_variant="mlp",
        backbone_variant="standard",
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 1.0,
        **kwargs  # Accept and ignore other PyTorch-specific params
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.mask_estimator_variant = mask_estimator_variant or "mlp"
        self.backbone_variant = backbone_variant or "standard"
        if self.backbone_variant not in {"standard", "value_residual", "siamese"}:
            raise ValueError(f"unknown backbone_variant: {self.backbone_variant!r}")
        self.num_stems = num_stems
        self.mlp_expansion_factor = mlp_expansion_factor

        # STFT parameters
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self._stft_transform = get_transform_mlx(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            window_fn="hann",
            window=None,
            periodic=True,
            center=True,
            normalized=stft_normalized,
        )
        # Optional chunked inference configuration (retained for checkpoint /
        # config compatibility; chunking itself is owned by the caller-side
        # backend, not this class -- see backends/mlx_backend.py).
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.freqs_per_bands = freqs_per_bands
        self.experimental_grouped_band_split = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT",
            default_value=False,
        )
        self.experimental_grouped_mask_estimator = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_MASK_ESTIMATOR",
            default_value=False,
        )
        self.experimental_compile_fullgraph = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_COMPILE_FULLGRAPH",
            default_value=False,
        )
        self._forward_model_compile_cache: dict[tuple[int, ...], object] = {}
        self._forward_model_compile_disabled: set[tuple[int, ...]] = set()

        # Verify frequency bands sum to expected number
        expected_freqs = stft_n_fft // 2 + 1
        assert sum(freqs_per_bands) == expected_freqs, \
            f"freqs_per_bands must sum to {expected_freqs}, got {sum(freqs_per_bands)}"

        transformer_kwargs = {
            "dim": dim,
            "heads": heads,
            "dim_head": dim_head,
            "attn_dropout": attn_dropout,
            "ff_dropout": ff_dropout,
            "ff_mult": mlp_expansion_factor,
            "norm_output": False,
        }

        # Use fast.rope directly in Attention layer (no need to create RoPE objects)
        # Pass a flag to indicate RoPE should be used
        rotary_embed = True  # Shared RoPE flag across all transformer branches

        # Build transformer layers with proper module registration
        self.depth = depth
        for i in range(depth):
            # Create transformers
            linear_tran = None
            if linear_transformer_depth > 0:
                # Keep RoPE consistent across all transformer branches.
                linear_tran = Transformer(
                    depth=linear_transformer_depth, rotary_embed=rotary_embed,
                    linear_attn=False, **transformer_kwargs,
                )

            if self.backbone_variant == "siamese":
                from .variants import SiameseTransformer

                time_tran = SiameseTransformer(depth=time_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)
                freq_tran = SiameseTransformer(depth=freq_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)
            elif self.backbone_variant == "value_residual":
                from .variants import ValueCaptureTransformer, ValueResidualTransformer

                transformer_cls = ValueCaptureTransformer if i == 0 else ValueResidualTransformer
                time_tran = transformer_cls(depth=time_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)
                freq_tran = transformer_cls(depth=freq_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)
            else:
                time_tran = Transformer(depth=time_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)
                freq_tran = Transformer(depth=freq_transformer_depth, rotary_embed=rotary_embed, **transformer_kwargs)

            # Create and register block
            setattr(self, f'layers_{i}', BSRoformerBlock(linear_tran, time_tran, freq_tran))

        self.final_norm = L2Norm(dim)

        # Band split with complex representation (2x for real/imag, stereo channels)
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            use_grouped=self.experimental_grouped_band_split,
        )

        # Mask estimators (one per stem) - stored as individual attributes.
        # NOT UPSTREAM: upstream always built the stock MaskEstimator and swallowed
        # any variant argument through **kwargs, so a checkpoint trained with a
        # different head constructed without error and computed the wrong thing.
        # Selection is explicit and owned by heads/__init__.py.
        from .heads import build_mask_estimator

        for i in range(num_stems):
            setattr(self, f'mask_estimators_{i}', build_mask_estimator(
                variant=self.mask_estimator_variant,
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                audio_channels=self.audio_channels,
                mlp_expansion_factor=mlp_expansion_factor,
                use_grouped=self.experimental_grouped_mask_estimator,
            ))

        if os.environ.get("MLX_ENABLE_COMPILE") == "1":
            # Compile only the transformer-heavy subgraph to maximize kernel fusion.
            self._forward_transformers = mx.compile(self._forward_transformers)

    def __call__(self, raw_audio):
        """
        Forward pass: raw audio -> STFT -> process -> apply masks -> iSTFT -> separated audio.

        This method mirrors the PyTorch BSRoformer implementation, handling the complete
        separation pipeline including STFT/iSTFT processing.

        Args:
            raw_audio: Input audio (batch, channels, time) or (batch, time)

        Returns:
            recon_audio: Separated audio (batch, num_stems, channels, time)
        """
        # Handle mono input
        if raw_audio.ndim == 2:
            raw_audio = mx.expand_dims(raw_audio, axis=1)  # (b, 1, t)

        batch_size, channels, time_samples = raw_audio.shape
        fixed_len_env = os.environ.get("MLX_FIXED_CHUNK_SAMPLES")
        if fixed_len_env:
            fixed_len = int(fixed_len_env)
            if time_samples > fixed_len:
                raise ValueError(f"Input length {time_samples} exceeds MLX_FIXED_CHUNK_SAMPLES={fixed_len}")
            if time_samples < fixed_len:
                pad_amount = fixed_len - time_samples
                raw_audio = mx.pad(raw_audio, [(0, 0), (0, 0), (0, pad_amount)])

        # Verify channel configuration
        if (self.stereo and channels != 2) or ((not self.stereo) and channels != 1):
            raise ValueError(f"Config mismatch: stereo={self.stereo} but input has {channels} channel(s)")

        # Reshape for STFT: (batch * channels, time)
        audio_flat = rearrange(raw_audio, "b c t -> (b c) t")


        # Batch STFT to avoid per-channel Python loops.
        with exact_zero_safe_rfft():
            stft_complex = self._stft_transform.stft(audio_flat)  # (b*c, F, N) complex
            mx.eval(stft_complex)
        stft_real = mx.stack([stft_complex.real, stft_complex.imag], axis=-1)  # (b*c, F, N, 2)

        # Reshape: First unpack (b*c) to (b, c), then merge (f, c) to (f*c)
        # This matches PyTorch: unpack_one then rearrange "b s f t c -> b (f s) t c"
        stft_repr = stft_real

        # Step 1: Reshape from (b*c, f, t, 2) to (b, c, f, t, 2)
        stft_repr = mx.reshape(stft_repr, (batch_size, channels, stft_repr.shape[1], stft_repr.shape[2], 2))

        # Step 2: Rearrange from (b, c, f, t, 2) to (b, f*c, t, 2)
        # Transpose to (b, f, c, t, 2) then reshape
        stft_repr = mx.transpose(stft_repr, (0, 2, 1, 3, 4))  # (b, f, c, t, 2)
        stft_repr = mx.reshape(stft_repr, (batch_size, stft_repr.shape[1] * channels, stft_repr.shape[3], 2))


        # Process through model to get masks
        masks = self._forward_model(stft_repr)

        # Before applying masks, add stem dimension to STFT (matching PyTorch)
        # stft_repr: (b, f*c, t, 2) -> (b, 1, f*c, t, 2)
        stft_repr_expanded = mx.expand_dims(stft_repr, axis=1)

        # Apply masks to STFT (complex multiplication)
        # Convert to complex representation for multiplication
        stft_complex = stft_repr_expanded[..., 0] + 1j * stft_repr_expanded[..., 1]  # (b, 1, f*c, t)
        mask_complex = masks[..., 0] + 1j * masks[..., 1]  # (b, n, f*c, t)

        # Apply masks via broadcasting: (b, 1, f*c, t) * (b, n, f*c, t) = (b, n, f*c, t)
        stft_masked = stft_complex * mask_complex  # (b, n, f*c, t)

        # Reshape for iSTFT: (b, n, f, c, t) -> (b*n*c, f, t)
        stft_masked = rearrange(stft_masked, "b n (f c) t -> (b n c) f t",
                               c=self.audio_channels)


        # Store original audio length for trimming
        original_length = raw_audio.shape[-1]

        # Batch iSTFT to avoid per-stem evaluation barriers.
        recon_audio = self._stft_transform.istft(
            stft_masked,
            length=original_length,
        )
        recon_audio = rearrange(recon_audio, "(b n c) t -> b n c t",
                               b=batch_size, n=self.num_stems, c=self.audio_channels)

        # Handle single stem case
        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 c t -> b c t")

        return recon_audio

    def _forward_transformers(self, x):
        """
        Process STFT representation through transformer stack.

        Args:
            x: Band-split features (batch, time, bands, dim)

        Returns:
            x: Transformer output (batch, time, bands, dim)
        """
        use_amp = os.environ.get("MLX_ENABLE_AMP") == "1"
        if use_amp:
            # Optional inference acceleration; keep STFT/ISTFT paths in float32.
            try:
                x = x.astype(mx.bfloat16)
            except Exception as exc:  # noqa: BLE001
                if not getattr(self, "_amp_warned", False) and os.environ.get("MLX_DEBUG") == "1":
                    print(f"[BSRoformerMLX] AMP disabled (bfloat16 unsupported): {exc}")
                    self._amp_warned = True
                use_amp = False

        if self.backbone_variant == "siamese":
            return self._forward_siamese(x, use_amp=use_amp)

        time_v_residual = None
        freq_v_residual = None

        # Apply transformer layers
        for i in range(self.depth):
            block = getattr(self, f'layers_{i}')

            # Access transformers from the block
            if block.has_linear:
                linear_transformer = block.linear_transformer
                time_transformer = block.time_transformer
                freq_transformer = block.freq_transformer

                # Linear attention (optional)
                x, ft_ps = pack([x], "b * d")
                x = linear_transformer(x)
                x, = unpack(x, ft_ps, "b * d")
            else:
                time_transformer = block.time_transformer
                freq_transformer = block.freq_transformer

            # Time transformer
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            if self.backbone_variant == "value_residual" and i > 0:
                x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)
                if time_v_residual is None:
                    time_v_residual = next_time_v_residual
            else:
                x = time_transformer(x)
            x, = unpack(x, ps, "* t d")

            # Frequency transformer
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            if self.backbone_variant == "value_residual" and i > 0:
                x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)
                if freq_v_residual is None:
                    freq_v_residual = next_freq_v_residual
            else:
                x = freq_transformer(x)
            x, = unpack(x, ps, "* f d")

        if use_amp:
            x = x.astype(mx.float32)

        # Final normalization
        x = self.final_norm(x)

        return x

    def _forward_siamese(self, x, *, use_amp=False):
        X = x
        Y = x
        layer_idx = 1
        for i in range(self.depth):
            block = getattr(self, f"layers_{i}")
            X = rearrange(X, "b t f d -> b f t d")
            Y = rearrange(Y, "b t f d -> b f t d")
            X, ps = pack([X], "* t d")
            Y, _ = pack([Y], "* t d")
            X, Y = block.time_transformer(X, Y, layer_idx)
            layer_idx += block.time_transformer.depth
            X, = unpack(X, ps, "* t d")
            Y, = unpack(Y, ps, "* t d")
            X = rearrange(X, "b f t d -> b t f d")
            Y = rearrange(Y, "b f t d -> b t f d")
            X, ps = pack([X], "* f d")
            Y, _ = pack([Y], "* f d")
            X, Y = block.freq_transformer(X, Y, layer_idx)
            layer_idx += block.freq_transformer.depth
            X, = unpack(X, ps, "* f d")
            Y, = unpack(Y, ps, "* f d")
        x = X + self.final_norm(Y)
        return x.astype(mx.float32) if use_amp else x

    def _estimate_masks(self, x):
        """
        Generate masks from transformer output (kept uncompiled).

        Args:
            x: Transformer output (batch, time, bands, dim)

        Returns:
            masks: Complex masks (batch, num_stems, freq*channels, time, 2)
        """
        masks = []
        for i in range(self.num_stems):
            estimator = getattr(self, f'mask_estimators_{i}')
            mask_output = estimator(x)
            masks.append(mask_output)
        masks = mx.stack(masks, axis=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
        return masks

    def _forward_model_impl(self, stft_repr):
        """
        Process STFT representation through transformer to generate masks.

        Args:
            stft_repr: STFT representation (batch, freq*channels, time, 2)

        Returns:
            masks: Complex masks (batch, num_stems, freq*channels, time, 2)
        """
        # Band split kept outside the compiled transformer stack.
        x = rearrange(stft_repr, "b f t c -> b t (f c)")
        x = self.band_split(x)
        x = self._forward_transformers(x)
        return self._estimate_masks(x)

    def _forward_model(self, stft_repr):
        """Forward-model wrapper with optional shape-keyed full-graph compile cache."""
        if not self.experimental_compile_fullgraph:
            return self._forward_model_impl(stft_repr)

        shape_key = tuple(int(v) for v in stft_repr.shape)
        if shape_key in self._forward_model_compile_disabled:
            return self._forward_model_impl(stft_repr)

        compiled_fn = self._forward_model_compile_cache.get(shape_key)
        if compiled_fn is None:
            compile_fn = getattr(mx, "compile", None)
            if not callable(compile_fn):
                self._forward_model_compile_disabled.add(shape_key)
                return self._forward_model_impl(stft_repr)

            def _compiled_forward(stft_in):
                return self._forward_model_impl(stft_in)

            try:
                compiled_fn = compile_fn(_compiled_forward, shapeless=False)
                self._forward_model_compile_cache[shape_key] = compiled_fn
            except Exception:  # noqa: BLE001
                self._forward_model_compile_disabled.add(shape_key)
                return self._forward_model_impl(stft_repr)

        try:
            return compiled_fn(stft_repr)
        except Exception:  # noqa: BLE001
            # Keep failures isolated by shape and fall back safely.
            self._forward_model_compile_disabled.add(shape_key)
            self._forward_model_compile_cache.pop(shape_key, None)
            return self._forward_model_impl(stft_repr)
