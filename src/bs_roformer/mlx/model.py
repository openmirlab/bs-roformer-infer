"""BSRoformerMLX -- the BS-RoFormer model architecture ported to Apple MLX.

Vendored (not depended on) because the upstream package this ships from,
`mlx-audio-separator`, bundles a full model catalog, a CLI, Demucs/VR/MDX
architectures, and a compiled `mlx-audio-io` extension that pins `mlx==0.31.2`
exactly -- installing it would drag all of that in for one model class. This
file is copied verbatim from upstream's BS-RoFormer module (which is already
BS-RoFormer-only; MelBand-Roformer lives in a sibling upstream file and was
never part of this one, so nothing needed stripping here). The only change
from upstream is this header; `from mlx_spectro import get_transform_mlx`
and `from packaging import version` are kept as real dependencies (both are
declared floors -- see pyproject.toml's `mlx` extra and base `dependencies`).

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/separator/models/roformer/bs_roformer.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.

Reads: mlx.core, mlx.nn, numpy, mlx_spectro (get_transform_mlx), packaging
"""

import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Optional, Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402
import numpy as np
from mlx_spectro import get_transform_mlx
from packaging import version

_USE_SAFE_SLICE_ACCUMULATION = version.parse(mx.__version__) >= version.parse("0.31.2")

# Helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d


def env_enabled(name: str, default_value: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default_value)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def batched_group_linear(x_group: mx.array, weights: mx.array, biases: mx.array | None = None) -> mx.array:
    """Apply per-group linear layers in a single batched matmul.

    Args:
        x_group: Shape (B, T, G, In)
        weights: Shape (G, Out, In)
        biases: Optional shape (G, Out)
    """
    out = mx.einsum("btgi,goi->btgo", x_group, weights)
    if biases is not None:
        out = out + biases[None, None, :, :]
    return out


# Use MLX's built-in nn.Sequential, which stores layers in self.layers list
# Weight keys need to use "layers.0", "layers.1", etc. to access them


# MLX-native replacements for einops operations
def pack(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """
    Pack tensors by flattening dimensions according to pattern.

    Patterns:
    - "b * d" : keep first and last dim, flatten middle dims
    - "* t d" : flatten leading dims, keep last two dims
    - "* f d" : flatten leading dims, keep last two dims
    """
    if len(tensors) == 1:
        x = tensors[0]
        original_shape = x.shape

        if pattern == "b * d":
            # Keep first and last dim, flatten middle
            # (b, f, t, d) -> (b, f*t, d)
            if len(x.shape) == 4:
                b, f, t, d = x.shape
                x = mx.reshape(x, (b, f * t, d))
            elif len(x.shape) == 3:
                # Already in the right shape
                pass
            return x, [original_shape]

        elif pattern == "* t d" or pattern == "* f d":
            # Flatten leading dims, keep last two
            # (b, f, t, d) -> (b*f, t, d)
            if len(x.shape) >= 3:
                *leading, t, d = x.shape
                leading_size = int(np.prod(leading))
                x = mx.reshape(x, (leading_size, t, d))
            return x, [original_shape]

        else:
            raise NotImplementedError(f"Pack pattern not implemented: {pattern}")
    else:
        # Stack multiple tensors
        packed = mx.stack(tensors, axis=0)
        shapes = [t.shape for t in tensors]
        return packed, shapes


def unpack(tensor: mx.array, shapes: List, pattern: str) -> List[mx.array]:  # noqa: UP006
    """
    Unpack tensor by restoring original shape.
    """
    if len(shapes) == 1:
        original_shape = shapes[0]
        x = mx.reshape(tensor, original_shape)
        return [x]
    else:
        # Unstack multiple tensors
        return [tensor[i] for i in range(len(shapes))]


def pack_one(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """Pack single tensor - alias for pack."""
    return pack(tensors, pattern)


def unpack_one(tensor: mx.array, shapes: List, pattern: str) -> mx.array:  # noqa: UP006
    """Unpack single tensor - returns first element."""
    return unpack(tensor, shapes, pattern)[0]


def rearrange(x: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    MLX-native implementation of einops rearrange for BS-Roformer patterns.
    Handles the specific rearrange patterns used in this model.
    """
    if "->" not in pattern:
        raise ValueError(f"Invalid pattern: {pattern}")

    input_pattern, output_pattern = pattern.split("->")
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()

    # Pattern: "b n (qkv h d) -> qkv b h n d" with qkv=3, h=heads
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h n d":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 1, 4))  # qkv, b, h, n, d
        return x

    # Pattern: "b n (qkv h d) -> qkv b h d n" with qkv=3, h=heads (for freq transformer)
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h d n":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 4, 1))  # qkv, b, h, d, n
        return x

    # Pattern: "b h n d -> b n h d"
    if input_pattern == "b h n d" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h d n -> b n h d"
    if input_pattern == "b h d n" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 3, 1, 2))

    # Pattern: "b n h d -> b h n d"
    if input_pattern == "b n h d" and output_pattern == "b h n d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h n d -> b n (h d)"
    if input_pattern == "b h n d" and output_pattern == "b n (h d)":
        b, h, n, d = x.shape
        return mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, n, h * d))

    # Pattern: "b h d n -> b n (h d)"
    if input_pattern == "b h d n" and output_pattern == "b n (h d)":
        b, h, d, n = x.shape
        return mx.reshape(mx.transpose(x, (0, 3, 1, 2)), (b, n, h * d))

    # Pattern: "b n h -> b h n 1"
    if input_pattern == "b n h" and output_pattern == "b h n 1":
        return mx.transpose(x, (0, 2, 1))[..., None]

    # Pattern: "b c t -> (b c) t"
    if input_pattern == "b c t" and output_pattern == "(b c) t":
        b, c, t = x.shape
        return mx.reshape(x, (b * c, t))

    # Pattern: "(b c) f t complex -> b (f c) t complex" with c=channels
    if input_pattern == "(b c) f t complex" and output_pattern == "b (f c) t complex":
        c = axes_lengths['c']
        bc, f, t, complex_dim = x.shape
        b = bc // c
        x = mx.reshape(x, (b, c, f, t, complex_dim))
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # b, f, c, t, complex
        x = mx.reshape(x, (b, f * c, t, complex_dim))
        return x

    # Pattern: "b n (f c) t -> (b n c) f t" with c=channels
    if input_pattern == "b n (f c) t" and output_pattern == "(b n c) f t":
        c = axes_lengths['c']
        b, n, fc, t = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, f, c, t))
        x = mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, c, f, t
        x = mx.reshape(x, (b * n * c, f, t))
        return x

    # Pattern: "(b n c) t -> b n c t" with b=batch, n=stems, c=channels
    if input_pattern == "(b n c) t" and output_pattern == "b n c t":
        b = axes_lengths['b']
        n = axes_lengths['n']
        c = axes_lengths['c']
        bnc, t = x.shape  # noqa: RUF059
        return mx.reshape(x, (b, n, c, t))

    # Pattern: "b 1 c t -> b c t"
    if input_pattern == "b 1 c t" and output_pattern == "b c t":
        return mx.squeeze(x, axis=1)

    # Pattern: "b f t c -> b t (f c)"
    if input_pattern == "b f t c" and output_pattern == "b t (f c)":
        b, f, t, c = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # b, t, f, c
        return mx.reshape(x, (b, t, f * c))

    # Pattern: "b t f d -> b f t d"
    if input_pattern == "b t f d" and output_pattern == "b f t d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b f t d -> b t f d"
    if input_pattern == "b f t d" and output_pattern == "b t f d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b n t (f c) -> b n f t c" with c=2
    if input_pattern == "b n t (f c)" and output_pattern == "b n f t c":
        c = axes_lengths['c']
        b, n, t, fc = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, t, f, c))
        return mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, f, t, c

    raise NotImplementedError(f"Rearrange pattern not implemented: {pattern}")


@contextmanager
def exact_zero_safe_rfft():
    """Route `mx.fft.rfft` through the CPU stream for one STFT. NOT upstream code.

    MLX 0.31.2's Metal rfft kernel packs two real FFTs into one complex FFT; in
    float32 that cancellation is not bit-exact, so a frame whose true value is
    exactly zero comes back as roughly 4.5e-07 instead of 0. That matters far more
    than its size suggests: `L2Norm` discards magnitude, and its eps of 1e-12 is
    five orders below the artifact, so the clamp never engages and pure numerical
    noise is normalized into a full-scale, essentially random feature vector --
    about a millionfold amplification. Time-axis attention then spreads that one
    corrupted frame across every position, which is why silence at the end of a
    chunk corrupts the output at the beginning.

    This is not a corner case: every track's final chunk is padded, and music has
    rests. Measured on the real checkpoint, a zero-padded chunk diverged from Torch
    by 1.455e-02 max abs; with this workaround, 2.012e-07 -- the same noise floor
    as a chunk with no silence at all.

    Raising `L2Norm`'s eps was considered and rejected: genuinely quiet audio has
    legitimate band norms in the same range, so it would trade this bug for a
    quieter one that only shows up on soft material.

    Caveat, stated rather than hidden: this swaps a module-level attribute, so it
    is not thread-safe. Inference here is single-threaded per session.

    Delete this once MLX's kernel is fixed. See brain/evidence.md section 14.
    """
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original


class L2Norm(nn.Module):
    """PyTorch-compatible norm: x / max(||x||, eps) * sqrt(dim) * weight."""

    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.weight = mx.ones((dim,))
        self.use_fast_norm = str(os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def __call__(self, x):
        if self.use_fast_norm:
            # Equivalent to L2 normalization with sqrt(dim) scaling.
            return mx.fast.rms_norm(x, self.weight, self.eps) * self.scale
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        denom = mx.maximum(norm, self.eps)
        return (x / denom) * self.scale * self.weight


class ExactGELU(nn.Module):
    """Exact GELU to match PyTorch's default (erf-based) implementation."""

    def __call__(self, x):
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


# Core Components using MLX built-ins

class FeedForward(nn.Module):
    """
    Feed-forward network with RMSNorm.
    Uses MLX built-in RMSNorm for optimal performance.
    Matches PyTorch structure with nn.Sequential for weight compatibility.
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)

        # Use MLX Sequential (stores layers in self.layers list)
        # Weights accessed as net.layers.0, net.layers.1, etc.
        self.net = nn.Sequential(
            L2Norm(dim),            # net.layers.0
            nn.Linear(dim, dim_inner),  # net.layers.1
            ExactGELU(),            # net.layers.2
            nn.Dropout(dropout),    # net.layers.3
            nn.Linear(dim_inner, dim),  # net.layers.4
            nn.Dropout(dropout)     # net.layers.5
        )

    def __call__(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention with rotary embeddings and gating.
    Uses MLX optimized components:
    - mlx.nn.RMSNorm for normalization
    - mlx.nn.RoPE for rotary position embeddings
    - mlx.core.fast.scaled_dot_product_attention for attention computation
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        # rotary_embed is now a boolean flag indicating whether to use RoPE
        self.use_rotary_embed = rotary_embed if isinstance(rotary_embed, bool) else (rotary_embed is not None)
        self.norm = L2Norm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)

        # Use MLX Sequential (stores layers in self.layers list)
        # Weights accessed as to_out.layers.0, to_out.layers.1
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),  # to_out.layers.0
            nn.Dropout(dropout)                      # to_out.layers.1
        )

    _fast_sdp_logged = False

    def __call__(self, x):
        x = self.norm(x)

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings if enabled
        if self.use_rotary_embed:
            # Use fast.rope which expects (batch, *, seq_len, dim_head)
            # Current shape is already (batch, heads, seq_len, dim_head) which is perfect
            # fast.rope with traditional=True matches PyTorch's RotaryEmbedding exactly
            q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)

        if os.environ.get("MLX_USE_FAST_SDP") == "1":
            if not Attention._fast_sdp_logged and os.environ.get("MLX_DEBUG") == "1":
                print("[BSRoformerMLX] Using mx.fast.scaled_dot_product_attention (MLX_USE_FAST_SDP=1)")
                Attention._fast_sdp_logged = True
            # Optional fast path; may change numerics vs PyTorch.
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # Manual attention to match PyTorch behavior; mx.fast kernel diverges in practice.
            attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            attn_scores = attn_scores - mx.max(attn_scores, axis=-1, keepdims=True)
            attn = mx.softmax(attn_scores, axis=-1)
            out = mx.matmul(attn, v)

        # Apply gating mechanism
        gates = self.to_gates(x)
        gates = mx.sigmoid(gates)
        gates = rearrange(gates, "b n h -> b h n 1")
        out = out * gates

        # Merge heads and project
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Linear attention variant for optional use.
    Based on: https://arxiv.org/abs/2106.09681
    """

    def __init__(self, dim, dim_head=32, heads=8, dropout=0.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.norm = L2Norm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        # Temperature parameter
        self.temperature = mx.ones((heads, 1, 1))

        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        b, n, _ = x.shape  # noqa: RUF059

        x = self.norm(x)

        # Project and rearrange
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h d n", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # L2 normalize
        q = q / mx.sqrt(mx.sum(q * q, axis=-2, keepdims=True) + 1e-8)
        k = k / mx.sqrt(mx.sum(k * k, axis=-2, keepdims=True) + 1e-8)

        # Apply temperature
        q = q * mx.exp(self.temperature)

        # Linear attention
        context = mx.matmul(k, v.transpose(0, 1, 3, 2))
        out = mx.matmul(q, context)

        # Rearrange and project
        out = rearrange(out, "b h d n -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feedforward."""
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def __call__(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        linear_attn=False
    ):
        super().__init__()
        self.depth = depth

        # Store as individual attributes for proper MLX registration
        for i in range(depth):
            if linear_attn:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed
                )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            setattr(self, f'layers_{i}', TransformerLayer(attn, ff))

        self.norm = L2Norm(dim) if norm_output else nn.Identity()

    def __call__(self, x):
        for i in range(self.depth):
            layer = getattr(self, f'layers_{i}')
            x = layer(x)

        return self.norm(x)


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

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):  # noqa: RUF007
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
        # Optional chunked inference configuration (used by separate_audio_chunked / separate)
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

        transformer_kwargs = dict(  # noqa: C408
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=mlp_expansion_factor,
            norm_output=False
        )

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


    def separate_audio_chunked(
        self,
        raw_audio: mx.array,
        *,
        sr: int = 44100,
        chunk_seconds: Optional[float] = None,  # noqa: UP045
        overlap_seconds: Optional[float] = None,  # noqa: UP045
        use_hann_window: bool = True,
        batch_hops: int = 8,
    ) -> mx.array:
        """
        Chunked, overlap-add inference for long audio.

        This is intentionally conservative for correctness: it runs the model on
        overlapping time-domain windows and recombines stems with weighted overlap-add.

        Args:
            raw_audio: (B, C, T) or (B, T) or (C, T) or (T,)
            sr: sample rate used to convert seconds -> samples.
            chunk_seconds / overlap_seconds: overrides for self.chunk_seconds/self.overlap_seconds.
            use_hann_window: if True, apply a Hann window during overlap-add.

        Returns:
            Same shape convention as __call__:
              - num_stems == 1: (B, C, T)
              - num_stems > 1:  (B, S, C, T)
        """
        # Normalize input shape to (B, C, T)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, None, :]
        elif raw_audio.ndim == 2:
            # Could be (C, T) or (B, T). Assume (C, T) if C in {1,2}
            if raw_audio.shape[0] in (1, 2):
                raw_audio = raw_audio[None, ...]
            else:
                raw_audio = raw_audio[:, None, :]
        elif raw_audio.ndim != 3:
            raise ValueError(f"Expected audio with 1-3 dims, got shape {tuple(raw_audio.shape)}")

        B, C, T = raw_audio.shape

        # Resolve chunk params
        chunk_s = float(self.chunk_seconds if chunk_seconds is None else chunk_seconds)
        overlap_s = float(self.overlap_seconds if overlap_seconds is None else overlap_seconds)

        chunk_len = int(round(chunk_s * sr))  # noqa: RUF046
        overlap_len = int(round(overlap_s * sr))  # noqa: RUF046
        if chunk_len <= 0:
            raise ValueError(f"chunk_seconds too small -> chunk_len={chunk_len}")
        if overlap_len < 0:
            raise ValueError(f"overlap_seconds must be >= 0, got {overlap_s}")
        if overlap_len >= chunk_len:
            raise ValueError(f"overlap ({overlap_len}) must be < chunk_len ({chunk_len})")

        hop_len = chunk_len - overlap_len

        # Pad on the right so we can take full chunks
        n_hops = int(math.ceil(max(T - overlap_len, 1) / hop_len))  # noqa: RUF046
        total_len = (n_hops - 1) * hop_len + chunk_len
        pad_len = total_len - T
        padded = mx.pad(raw_audio, [(0, 0), (0, 0), (0, pad_len)]) if pad_len > 0 else raw_audio

        # Overlap-add weights
        if use_hann_window and chunk_len > 1:
            w = np.hanning(chunk_len).astype(np.float32)
        else:
            w = np.ones((chunk_len,), dtype=np.float32)
        w_mx = mx.array(w, dtype=mx.float32)
        w_view_single = w_mx.reshape(1, 1, -1)
        w_view_multi = w_mx.reshape(1, 1, 1, -1)

        # Keep accumulation on-device to avoid per-batch host transfers.
        if self.num_stems == 1:
            out_acc = mx.zeros((B, C, total_len), dtype=mx.float32)
            w_acc = mx.zeros((1, 1, total_len), dtype=mx.float32)
        else:
            out_acc = mx.zeros((B, self.num_stems, C, total_len), dtype=mx.float32)
            w_acc = mx.zeros((1, 1, 1, total_len), dtype=mx.float32)

        batch_hops = int(batch_hops)
        if batch_hops <= 0:
            raise ValueError(f"batch_hops must be >= 1, got {batch_hops}")
        eval_flush_interval = max(8, batch_hops * 2)
        pending_updates = 0

        # Precompute all starts to avoid recomputing in the loop
        starts = [hop * hop_len for hop in range(n_hops)]
        arange_chunk = mx.arange(chunk_len, dtype=mx.int32)
        all_starts_mx = mx.array(starts, dtype=mx.int32)
        all_gather_idx = all_starts_mx[:, None] + arange_chunk[None, :]
        use_gather_batching = env_enabled(
            "MLX_AUDIO_SEPARATOR_ROFORMER_CHUNK_GATHER_BATCHING",
            default_value=False,
        )

        for i in range(0, n_hops, batch_hops):
            hops = list(range(i, min(i + batch_hops, n_hops)))
            H = len(hops)

            # Stack chunks along batch dimension: (B*H, C, L)
            if use_gather_batching:
                try:
                    gather_idx = all_gather_idx[i : i + H]
                    chunk_batch = mx.transpose(padded[:, :, gather_idx], (2, 0, 1, 3))
                    chunk_batch = chunk_batch.reshape(H * B, C, chunk_len)
                except Exception:  # noqa: BLE001
                    chunk_list = [padded[..., starts[h]:starts[h] + chunk_len] for h in hops]
                    chunk_batch = mx.concatenate(chunk_list, axis=0)
            else:
                chunk_list = [padded[..., starts[h]:starts[h] + chunk_len] for h in hops]
                chunk_batch = mx.concatenate(chunk_list, axis=0)

            # Run model once for the whole batch
            batch_out = self(chunk_batch)
            if self.num_stems == 1:
                # batch_out: (B*H, C, L) -> (H, B, C, L)
                batch_out = batch_out.reshape(H, B, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_update = batch_out[j] * w_view_single
                    if _USE_SAFE_SLICE_ACCUMULATION:
                        start_mx = mx.array([start])
                        out_acc = mx.slice_update(
                            out_acc,
                            out_acc[..., start:end] + out_update,
                            start_mx,
                            axes=(out_acc.ndim - 1,),
                        )
                        w_acc = mx.slice_update(
                            w_acc,
                            w_acc[..., start:end] + w_view_single,
                            start_mx,
                            axes=(w_acc.ndim - 1,),
                        )
                    else:
                        out_acc = out_acc.at[..., start:end].add(out_update)
                        w_acc = w_acc.at[..., start:end].add(w_view_single)
            else:
                # batch_out: (B*H, S, C, L) -> (H, B, S, C, L)
                batch_out = batch_out.reshape(H, B, self.num_stems, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_update = batch_out[j] * w_view_multi
                    if _USE_SAFE_SLICE_ACCUMULATION:
                        start_mx = mx.array([start])
                        out_acc = mx.slice_update(
                            out_acc,
                            out_acc[..., start:end] + out_update,
                            start_mx,
                            axes=(out_acc.ndim - 1,),
                        )
                        w_acc = mx.slice_update(
                            w_acc,
                            w_acc[..., start:end] + w_view_multi,
                            start_mx,
                            axes=(w_acc.ndim - 1,),
                        )
                    else:
                        out_acc = out_acc.at[..., start:end].add(out_update)
                        w_acc = w_acc.at[..., start:end].add(w_view_multi)

            pending_updates += H
            if pending_updates >= eval_flush_interval:
                mx.eval(out_acc, w_acc)
                pending_updates = 0

        mx.eval(out_acc, w_acc)
        out_acc = out_acc / mx.maximum(w_acc, 1e-8)
        out_acc = out_acc[..., :T]
        mx.eval(out_acc)
        return out_acc

    def separate(self, wav: mx.array, *, sr: int = 44100) -> mx.array:
        """
        Convenience wrapper for inference.

        Accepts:
          - (T,)
          - (C, T)
          - (B, T)
          - (B, C, T)

        Returns:
          - if num_stems == 1: (C, T) or (B, C, T)
          - if num_stems > 1:  (S, C, T) or (B, S, C, T)
        """
        if wav.ndim == 1:
            wav_bct = wav[None, None, :]
            squeeze_b = True
        elif wav.ndim == 2:
            if wav.shape[0] in (1, 2):
                wav_bct = wav[None, ...]
                squeeze_b = True
            else:
                wav_bct = wav[:, None, :]
                squeeze_b = False
        elif wav.ndim == 3:
            wav_bct = wav
            squeeze_b = (wav.shape[0] == 1)
        else:
            raise ValueError(f"Expected wav with 1-3 dims, got shape {tuple(wav.shape)}")

        chunk_len = int(round(self.chunk_seconds * sr))  # noqa: RUF046
        if wav_bct.shape[-1] > chunk_len:
            out = self.separate_audio_chunked(wav_bct, sr=sr)
        else:
            out = self(wav_bct)

        if squeeze_b:
            out = out[0]

        return out


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
            x = time_transformer(x)
            x, = unpack(x, ps, "* t d")

            # Frequency transformer
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            x, = unpack(x, ps, "* f d")

        if use_amp:
            x = x.astype(mx.float32)

        # Final normalization
        x = self.final_norm(x)

        return x

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


def create_compiled_model(*args, **kwargs):
    """
    Create a BS-Roformer model with compiled forward pass for optimal performance.

    Usage:
        model = create_compiled_model(dim=128, depth=12, ...)
        masks = model(stft_repr)
    """
    model = BSRoformerMLX(*args, **kwargs)

    return model
