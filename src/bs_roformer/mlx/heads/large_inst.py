"""LargeInstMaskEstimator -- MLX port of the Large-Inst mask-estimator head.

Mirrors ``bs_roformer.large_inst.LargeInstMaskEstimator``: four alternating
time/frequency Transformer pairs (depth 1 each, heads/dim_head hard-coded to
8/64 exactly as the Torch original hard-codes them -- the constructor
signature here intentionally has no way to override that, matching the Torch
call site) run before the per-band mask MLP. Built mostly from blocks already
owned by the trunk (``..attention.L2Norm``, ``..bands.MLP``,
``..attention.FeedForward``, ``..attention.TransformerLayer``, the einops-lite
``..ops.pack``/``..ops.unpack``/``..ops.rearrange`` helpers) rather than
reimplementing them.

The one block NOT reused as-is is ``..attention.Attention``: this checkpoint's
own two ``RotaryEmbedding`` modules (one shared across the four time-axis
blocks, one shared across the four freq-axis blocks) have ``learned_freq``
drift baked in by training -- their saved ``rotary_embed.freqs`` do NOT match
the theta=10000 default that ``..attention.Attention`` hard-codes for
``rotary_embed=True``. Verified against the real ``bs_large_v2_inst.ckpt``
checkpoint: the stored per-axis freqs differ from the theoretical theta=10000
formula by up to ~0.0088 and are not even monotonic (a fixed formula can
never go negative), which only a trained parameter explains. Reusing
``..attention.Attention`` unmodified for this head compounds that mismatch
across 8 attention layers into a >1.0 max-abs-error output (measured while
porting this file -- see the parity numbers reported alongside it).
``_LearnedRopeAttention`` below subclasses ``..attention.Attention`` (so it
inherits its exact weight-compatible submodule layout: ``norm``, ``to_qkv``,
``to_gates``, ``to_out``) and overrides only ``__call__`` to rotate Q/K with
an explicit loaded ``rope_freqs`` array via ``mx.fast.rope(freqs=...)``
instead of ``base=10000.0``. One further non-obvious point, confirmed
empirically rather than assumed: MLX's ``mx.fast.rope(freqs=...)`` expects
the *reciprocal* of what ``rotary_embedding_torch`` stores as
``RotaryEmbedding.freqs`` (an angular frequency per rotated dimension pair)
-- passing the checkpoint's ``rotary_embed.freqs`` straight through, without
inverting it, silently produces a wrong-but-plausible-shaped output (measured
~0.33 max abs error on a single depth-1 block using the real checkpoint's
learned freqs; ~5.0 on a standalone theta=10000 sanity check with known-good
values). ``rope_freqs`` is stored here in torch's native convention and
inverted at call time, so the loaded weight matches the checkpoint verbatim
and the inversion stays local to this file.

Reads: ..attention.Attention, ..attention.L2Norm, ..attention.FeedForward,
..attention.TransformerLayer, ..bands.MLP, ..ops.pack, ..ops.unpack,
..ops.rearrange, mlx.core, mlx.nn
"""

import os
from typing import Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from ..attention import Attention, FeedForward, L2Norm, TransformerLayer
from ..bands import MLP
from ..ops import pack, rearrange, unpack


class _LearnedRopeAttention(Attention):
    """``..attention.Attention`` with this checkpoint's own learned RoPE frequencies.

    See the module docstring for why this cannot just be ``Attention(...,
    rotary_embed=True)``. Everything except the RoPE call is inherited
    unchanged, so the weight-mapping for ``norm``/``to_qkv``/``to_gates``/
    ``to_out`` is identical to a plain ``Attention`` -- only ``rope_freqs``
    (shape ``(dim_head // 2,)``, loaded verbatim from the checkpoint's
    ``rotary_embed.freqs``) is new.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, rotary_embed=True)
        self.rope_freqs = mx.ones((dim_head // 2,))

    def __call__(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # mx.fast.rope's `freqs` is the reciprocal of rotary_embedding_torch's
        # `RotaryEmbedding.freqs` -- see module docstring; confirmed empirically.
        inv_freqs = 1.0 / self.rope_freqs
        q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=None, scale=1.0, offset=0, freqs=inv_freqs)
        k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=None, scale=1.0, offset=0, freqs=inv_freqs)

        if os.environ.get("MLX_USE_FAST_SDP") == "1":
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            attn_scores = attn_scores - mx.max(attn_scores, axis=-1, keepdims=True)
            attn = mx.softmax(attn_scores, axis=-1)
            out = mx.matmul(attn, v)

        gates = self.to_gates(x)
        gates = mx.sigmoid(gates)
        gates = rearrange(gates, "b n h -> b h n 1")
        out = out * gates

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _LargeInstPair(nn.Module):
    """One Large-Inst block: a time-axis TransformerLayer, then a frequency-axis one.

    Torch's depth is always 1 for these four blocks (hard-coded, independent
    of the mask MLP's own ``depth``), so this holds a single
    ``TransformerLayer`` per axis rather than a full ``..attention.Transformer``
    depth-loop wrapper.
    """

    def __init__(self, time_layer: TransformerLayer, freq_layer: TransformerLayer):
        super().__init__()
        self.time_transformer = time_layer
        self.freq_transformer = freq_layer


def _make_transformer_layer(dim: int) -> TransformerLayer:
    attn = _LearnedRopeAttention(dim=dim, heads=8, dim_head=64, dropout=0.0)
    ff = FeedForward(dim=dim, mult=4, dropout=0.0)
    return TransformerLayer(attn, ff)


class LargeInstMaskEstimator(nn.Module):
    """MLX port of pcunwa's Large-Inst mask-estimator head.

    Signature matches the Torch constructor call site
    (``bs_roformer.py``'s ``_create_mask_estimator``) exactly: ``dim``,
    ``dim_inputs``, ``depth`` (the per-band mask MLP depth), and
    ``mlp_expansion_factor``. The four time/freq Transformer pairs are not
    parameterized by ``depth`` -- Torch hard-codes them to one layer each,
    independent of the mask MLP depth, and this mirrors that.
    """

    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],  # noqa: UP006
        depth,
        mlp_expansion_factor=4,
    ):
        super().__init__()

        self.dim_inputs = dim_inputs
        self.num_bands = len(dim_inputs)
        dim_hidden = dim * mlp_expansion_factor

        self._mlp_module_names: list[str] = []
        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_freqs_{i}"
            setattr(self, module_name, MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth))
            self._mlp_module_names.append(module_name)

        for i in range(4):
            setattr(
                self,
                f"layers_{i}",
                _LargeInstPair(_make_transformer_layer(dim), _make_transformer_layer(dim)),
            )

        self.norm = L2Norm(dim)

    def __call__(self, x):
        for i in range(4):
            pair = getattr(self, f"layers_{i}")

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = pair.time_transformer(x)
            (x,) = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = pair.freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        x = self.norm(x)

        outs = []
        for i in range(self.num_bands):
            mlp = getattr(self, self._mlp_module_names[i])
            band_features = x[..., i, :]
            freq_out_before_glu = mlp(band_features)
            values, gates = mx.split(freq_out_before_glu, 2, axis=-1)
            outs.append(values * mx.sigmoid(gates))

        return mx.concatenate(outs, axis=-1)


__all__ = ["LargeInstMaskEstimator"]
