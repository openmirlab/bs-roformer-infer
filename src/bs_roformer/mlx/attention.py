"""Attention and transformer primitives shared by the BS-RoFormer trunk and heads.

`L2Norm` is the PyTorch-compatible normalization (`x / max(||x||, eps) *
sqrt(dim) * weight`) every block here and in `bands.py` is built on; its eps of
1e-12 is exactly what makes `rfft_guard.exact_zero_safe_rfft` load-bearing (see
that module's docstring). `Attention` implements RoPE + gated multi-head
attention with two numerically distinct paths -- `mx.fast.scaled_dot_product_attention`
(opt-in via `MLX_USE_FAST_SDP=1`, faster but not guaranteed to match PyTorch bit
for bit) and a manual matmul/softmax path used by default because it was measured
to track PyTorch more closely. `LinearAttention` and the `Transformer`
constructor's `linear_attn` flag are upstream surface that is currently
unreachable: every call site passes `linear_attn=False`, so `Transformer` always
builds `Attention`, never `LinearAttention` -- kept as-is (upstream-shaped, not
disproven safe to remove) rather than pruned in this pass.

Reads: .ops (rearrange)
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from .ops import rearrange


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
        _b, _n, _ = x.shape

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
