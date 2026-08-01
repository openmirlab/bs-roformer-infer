"""MLX HyperACE v1 head for the original pcunwa checkpoint variation.

Reads: .hyperace (Backbone, Decoder, HyperACE, DSConvBlock, _resize_bilinear),
..bands.MLP, mlx.core, mlx.nn
"""

from __future__ import annotations

from typing import Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from ..bands import MLP
from .hyperace import Backbone, DSConvBlock, Decoder, HyperACE, _resize_bilinear


class FreqPixelShuffleV1(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = DSConvBlock(in_channels, out_channels * scale, k=3, s=1, p=1)

    def __call__(self, x):
        x = self.conv(x)
        b, h, w, channels = x.shape
        out_channels = channels // self.scale
        x = mx.reshape(x, (b, h, w, out_channels, self.scale))
        x = mx.transpose(x, (0, 1, 2, 4, 3))
        return mx.reshape(x, (b, h, w * self.scale, out_channels))


class ProgressiveUpsampleHeadV1(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025):
        super().__init__()
        c = in_channels
        self.block1 = FreqPixelShuffleV1(c, c, scale=2)
        self.block2 = FreqPixelShuffleV1(c, c // 2, scale=2)
        self.block3 = FreqPixelShuffleV1(c // 2, c // 2, scale=2)
        self.block4 = FreqPixelShuffleV1(c // 2, c // 4, scale=2)
        self.final_conv = nn.Conv2d(c // 4, out_channels, kernel_size=1, bias=False)
        self.target_bins = target_bins

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if x.shape[2] != self.target_bins:
            x = _resize_bilinear(x, x.shape[1], self.target_bins)
        return self.final_conv(x)


class SegmModelV1(nn.Module):
    def __init__(self, in_bands=62, in_dim=256, out_bins=1025, out_channels=4, base_channels=64, base_depth=2, num_hyperedges=16, num_heads=8):
        super().__init__()
        self.backbone = Backbone(in_channels=in_dim, base_channels=base_channels, base_depth=base_depth)
        enc_channels = self.backbone.out_channels
        _, _, c4, _ = enc_channels
        self.hyperace = HyperACE(enc_channels, c4, num_hyperedges, num_heads, k=3, low_order_depth=2)
        self.decoder = Decoder(enc_channels, c4, list(enc_channels))
        self.upsample_head = ProgressiveUpsampleHeadV1(enc_channels[0], out_channels, out_bins)

    def __call__(self, x):
        height = x.shape[1]
        enc_feats = self.backbone(x)
        h_ace_feats = self.hyperace(enc_feats)
        dec_feat = self.decoder(enc_feats, h_ace_feats)
        return self.upsample_head(_resize_bilinear(dec_feat, height, dec_feat.shape[2]))


class HyperACEV1MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, audio_channels=2, mlp_expansion_factor=4):  # noqa: UP006
        super().__init__()
        self.num_bands = len(dim_inputs)
        dim_hidden = dim * mlp_expansion_factor
        for i, dim_in in enumerate(dim_inputs):
            setattr(self, f"to_freqs_{i}", MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth))
        self.segm = SegmModelV1(
            in_bands=self.num_bands,
            in_dim=dim,
            out_bins=sum(dim_inputs) // (2 * audio_channels),
            out_channels=2 * audio_channels,
        )

    def __call__(self, x):
        y = self.segm(x)
        y = mx.reshape(y, (y.shape[0], y.shape[1], -1))
        outs = []
        for i in range(self.num_bands):
            raw = getattr(self, f"to_freqs_{i}")(x[..., i, :])
            values, gates = mx.split(raw, 2, axis=-1)
            outs.append(values * mx.sigmoid(gates))
        return mx.concatenate(outs, axis=-1) + y


__all__ = ["HyperACEV1MaskEstimator"]
