"""MLX trunk variations for non-standard BS-RoFormer checkpoints.

Reads: .attention (Attention, FeedForward, L2Norm, TransformerLayer), mlx.core,
mlx.nn
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from .attention import Attention, FeedForward, L2Norm, TransformerLayer


class ValueResidualAttention(Attention):
    """Attention with the learned value-residual interpolation used by Torch."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None):
        super().__init__(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, rotary_embed=rotary_embed)
        self.to_value_residual_mix = nn.Linear(dim, heads)

    def __call__(self, x, value_residual=None):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = mx.reshape(qkv, (qkv.shape[0], qkv.shape[1], 3, self.heads, self.dim_head))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        original_v = v
        if value_residual is not None:
            mix = mx.sigmoid(self.to_value_residual_mix(x))
            mix = mx.transpose(mix, (0, 2, 1))[..., None]
            v = v + (value_residual - v) * mix
        if self.use_rotary_embed:
            q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
        if os.environ.get("MLX_USE_FAST_SDP") == "1":
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            out = mx.matmul(mx.softmax(scores, axis=-1), v)
        gates = mx.transpose(mx.sigmoid(self.to_gates(x)), (0, 2, 1))[..., None]
        out = mx.transpose(out * gates, (0, 2, 1, 3))
        out = mx.reshape(out, (out.shape[0], out.shape[1], -1))
        return self.to_out(out), original_v


class ValueResidualTransformer(nn.Module):
    """Transformer stack that returns its first attention value stream."""

    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, norm_output=True, rotary_embed=None):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            setattr(self, f"layers_{i}", TransformerLayer(
                ValueResidualAttention(dim, heads, dim_head, attn_dropout, rotary_embed),
                FeedForward(dim, ff_mult, ff_dropout),
            ))
        self.norm = L2Norm(dim) if norm_output else nn.Identity()

    def __call__(self, x, value_residual=None):
        first_values = None
        for i in range(self.depth):
            layer = getattr(self, f"layers_{i}")
            attn_out, next_values = layer.attn(x, value_residual=value_residual)
            x = x + attn_out
            if first_values is None:
                first_values = next_values
            x = x + layer.ff(x)
        return self.norm(x), first_values


class ValueCaptureAttention(Attention):
    """Stock attention that additionally returns its unmodified value stream."""

    def __call__(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = mx.reshape(qkv, (qkv.shape[0], qkv.shape[1], 3, self.heads, self.dim_head))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rotary_embed:
            q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=10000.0, scale=1.0, offset=0)
        if os.environ.get("MLX_USE_FAST_SDP") == "1":
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            out = mx.matmul(mx.softmax(scores, axis=-1), v)
        gates = mx.transpose(mx.sigmoid(self.to_gates(x)), (0, 2, 1))[..., None]
        out = mx.transpose(out * gates, (0, 2, 1, 3))
        out = mx.reshape(out, (out.shape[0], out.shape[1], -1))
        return self.to_out(out), v


class ValueCaptureTransformer(nn.Module):
    """First value-residual stack, which has no learned mixing projection."""

    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, norm_output=True, rotary_embed=None):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            setattr(self, f"layers_{i}", TransformerLayer(
                ValueCaptureAttention(dim, heads, dim_head, attn_dropout, rotary_embed),
                FeedForward(dim, ff_mult, ff_dropout),
            ))
        self.norm = L2Norm(dim) if norm_output else nn.Identity()

    def __call__(self, x):
        first_values = None
        for i in range(self.depth):
            layer = getattr(self, f"layers_{i}")
            attn_out, next_values = layer.attn(x)
            x = x + attn_out
            if first_values is None:
                first_values = next_values
            x = x + layer.ff(x)
        return self.norm(x), first_values


class SiameseTransformer(nn.Module):
    """Two-stream transformer matching the Torch Siamese parameter layout."""

    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, norm_output=True, rotary_embed=None):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            setattr(self, f"layers_{i}", TransformerLayer(
                Attention(dim, heads, dim_head, attn_dropout, rotary_embed),
                FeedForward(dim, ff_mult, ff_dropout),
            ))
            for stem in ("ln_y_attn", "ln_x_attn", "ln_y_mlp", "ln_x_mlp"):
                setattr(self, f"{stem}_{i}", L2Norm(dim))
            setattr(self, f"y_attn_{i}", mx.ones((dim,)))
        self.norm = L2Norm(dim) if norm_output else nn.Identity()

    def __call__(self, x, y, layer_idx_start=1):
        for i in range(self.depth):
            layer = getattr(self, f"layers_{i}")
            coupling = 1.0 / math.sqrt(layer_idx_start + i) + 1.0
            mixed = x * getattr(self, f"y_attn_{i}") + getattr(self, f"ln_y_attn_{i}")(y)
            attn_out = layer.attn(mixed)
            y = y + attn_out
            x = getattr(self, f"ln_x_attn_{i}")(x + attn_out * coupling)
            ff_out = layer.ff(x + getattr(self, f"ln_y_mlp_{i}")(y))
            y = y + ff_out
            x = getattr(self, f"ln_x_mlp_{i}")(x + ff_out * coupling)
        return x, y


__all__ = ["SiameseTransformer", "ValueCaptureTransformer", "ValueResidualTransformer"]
