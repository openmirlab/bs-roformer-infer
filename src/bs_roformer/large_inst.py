"""Large-Inst mask-estimator variation for BS-RoFormer checkpoints.

pcunwa's Large-Inst checkpoint keeps the standard BS-RoFormer trunk but inserts
four alternating time/frequency Transformer pairs before the per-band mask MLP.
The published state_dict stores those weights under ``mask_estimators.*.layers``;
this module implements only that inference head so the rest of the package can
continue selecting variants through ``mask_estimator_variant``.

Reads: .bs_roformer.Attention, .bs_roformer.FeedForward, .bs_roformer.MLP,
.bs_roformer.RMSNorm, rotary_embedding_torch.RotaryEmbedding, einops, torch
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList

from beartype import beartype
from einops import pack, rearrange, unpack
from rotary_embedding_torch import RotaryEmbedding


class _TensorTransformer(Module):
    """Checkpoint-compatible one-layer Transformer that returns only the tensor."""

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
        flash_attn=True,
    ):
        super().__init__()

        from .bs_roformer import Attention, FeedForward, RMSNorm

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            attn_out, _ = attn(x)
            x = x + attn_out
            x = x + ff(x)

        return self.norm(x)


class LargeInstMaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        mlp_expansion_factor=4,
    ):
        super().__init__()

        from .bs_roformer import MLP, RMSNorm

        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            self.to_freqs.append(mlp)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=8,
            dim_head=64,
            attn_dropout=0.0,
            ff_dropout=0.0,
            flash_attn=True,
            norm_output=False,
        )

        time_rotary_embed = RotaryEmbedding(dim=64)
        freq_rotary_embed = RotaryEmbedding(dim=64)

        for _ in range(4):
            self.layers.append(
                ModuleList(
                    [
                        _TensorTransformer(depth=1, rotary_embed=time_rotary_embed, **transformer_kwargs),
                        _TensorTransformer(depth=1, rotary_embed=freq_rotary_embed, **transformer_kwargs),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x):
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        x = self.norm(x)
        x = x.unbind(dim=-2)

        outs = []
        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


__all__ = ["LargeInstMaskEstimator"]
