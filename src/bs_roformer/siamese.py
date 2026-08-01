"""Clean-room Siamese transformer trunk for the pcunwa BS checkpoint family.

The experimental Siamese checkpoints keep the normal BS-RoFormer STFT, band
split, and mask-estimator boundary, but replace each axial Transformer with a
two-stream block. This module uses the package's existing attention, feed-forward,
and normalization primitives so the published parameter layout remains loadable
without importing the upstream research repository.

Reads: .bs_roformer.Attention, .bs_roformer.FeedForward, .bs_roformer.RMSNorm,
torch
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList

from .bs_roformer import Attention, FeedForward, RMSNorm


class SiameseTransformer(Module):
    """Couple an X stream and a Y stream through attention and MLP updates."""

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
        self.layers = ModuleList([])
        self.ln_y_attn = ModuleList([])
        self.ln_x_attn = ModuleList([])
        self.ln_y_mlp = ModuleList([])
        self.ln_x_mlp = ModuleList([])
        self.y_attn = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))
            self.ln_y_attn.append(RMSNorm(dim))
            self.ln_x_attn.append(RMSNorm(dim))
            self.ln_y_mlp.append(RMSNorm(dim))
            self.ln_x_mlp.append(RMSNorm(dim))
            self.y_attn.append(nn.Parameter(torch.ones(dim)))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, X, Y, layer_idx_start=1):
        for i, (attn, ff) in enumerate(self.layers):
            layer_index = layer_idx_start + i
            coupling = 1.0 / (layer_index ** 0.5) + 1.0

            mixed_a = X * self.y_attn[i] + self.ln_y_attn[i](Y)
            # The shared Attention also returns a value-residual tensor for
            # standard trunks; Siamese uses only its projected output.
            attn_out, _ = attn(mixed_a)
            Y = Y + attn_out
            X = self.ln_x_attn[i](X + attn_out * coupling)

            ff_out = ff(X + self.ln_y_mlp[i](Y))
            Y = Y + ff_out
            X = self.ln_x_mlp[i](X + ff_out * coupling)

        return X, Y


__all__ = ["SiameseTransformer"]
