"""HyperACE v1 mask-estimator variation for the pcunwa BS checkpoints.

HyperACE v1 shares the local HyperACE backbone and hypergraph blocks with the
v2 implementation, but its published head has a different channel schedule and
does not contain the v2 TFC-TDF refinement blocks. Keeping the head separate is
necessary for strict checkpoint loading: the two versions have different module
trees and tensor shapes.

Reads: .hyperace.Backbone, .hyperace.HyperACE, .hyperace.Decoder, torch, einops
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from einops import rearrange

from .hyperace import Backbone, Decoder, HyperACE, _mlp


class FreqPixelShuffleV1(nn.Module):
    """Frequency-only pixel shuffle used by the original HyperACE head."""

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        # Keep the upstream module names (conv + act) for strict loading.
        from .hyperace import DSConv
        self.conv = DSConv(in_channels, out_channels * scale, k=3, s=1, p=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        batch, channels, height, width = x.shape
        out_channels = channels // self.scale
        x = x.view(batch, out_channels, self.scale, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch, out_channels, height, width * self.scale)
        return x


class ProgressiveUpsampleHeadV1(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025):
        super().__init__()
        self.target_bins = target_bins
        c = in_channels
        self.block1 = FreqPixelShuffleV1(c, c, scale=2)
        self.block2 = FreqPixelShuffleV1(c, c // 2, scale=2)
        self.block3 = FreqPixelShuffleV1(c // 2, c // 2, scale=2)
        self.block4 = FreqPixelShuffleV1(c // 2, c // 4, scale=2)
        self.final_conv = nn.Conv2d(c // 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if x.shape[-1] != self.target_bins:
            x = F.interpolate(
                x,
                size=(x.shape[2], self.target_bins),
                mode="bilinear",
                align_corners=False,
            )
        return self.final_conv(x)


class SegmModelV1(nn.Module):
    def __init__(
        self,
        in_bands=62,
        in_dim=256,
        out_bins=1025,
        out_channels=4,
        base_channels=64,
        base_depth=2,
        num_hyperedges=16,
        num_heads=8,
    ):
        super().__init__()
        self.backbone = Backbone(
            in_channels=in_dim,
            base_channels=base_channels,
            base_depth=base_depth,
        )
        enc_channels = self.backbone.out_channels
        _, _, c4, _ = enc_channels
        self.hyperace = HyperACE(
            enc_channels,
            c4,
            num_hyperedges,
            num_heads,
            k=3,
            low_order_depth=2,
        )
        self.decoder = Decoder(enc_channels, c4, list(enc_channels))
        self.upsample_head = ProgressiveUpsampleHeadV1(
            in_channels=enc_channels[0],
            out_channels=out_channels,
            target_bins=out_bins,
        )

    def forward(self, x):
        height = x.shape[2]
        enc_feats = self.backbone(x)
        h_ace_feats = self.hyperace(enc_feats)
        dec_feat = self.decoder(enc_feats, h_ace_feats)
        dec_feat = F.interpolate(
            dec_feat,
            size=(height, dec_feat.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return self.upsample_head(dec_feat)


class HyperACEV1MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        audio_channels=2,
        mlp_expansion_factor=4,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor
        for dim_in in dim_inputs:
            self.to_freqs.append(nn.Sequential(
                _mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            ))

        out_channels = 2 * audio_channels
        self.segm = SegmModelV1(
            in_bands=len(dim_inputs),
            in_dim=dim,
            out_bins=sum(dim_inputs) // out_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        y = rearrange(x, "b t f c -> b c t f")
        y = self.segm(y)
        y = rearrange(y, "b c t f -> b t (f c)")
        outs = [mlp(features) for features, mlp in zip(x.unbind(dim=-2), self.to_freqs)]
        return torch.cat(outs, dim=-1) + y


__all__ = ["HyperACEV1MaskEstimator"]
