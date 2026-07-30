"""HyperACEMaskEstimator -- MLX port of the HyperACE mask-estimator head.

Mirrors ``bs_roformer.hyperace.HyperACEMaskEstimator``: a per-band GLU-MLP
mask path (identical in shape to the stock ``MaskEstimator``, reused via
``..model.MLP`` so it needs no new weight-mapping rule) added to the output of
``SegmModel``, a small conv/hypergraph segmentation network (Backbone ->
HyperACE hypergraph fusion -> Decoder -> ProgressiveUpsampleHead) that treats
the band-split trunk output as an image: time is height, band index is width,
feature dim is channels. Two layout facts drive every class below:

1. Torch's conv stack is NCHW; MLX's ``nn.Conv2d``/``nn.InstanceNorm`` are
   NHWC. The trunk already hands this head ``x`` shaped ``(b, t, f, dim)`` --
   exactly NHWC with H=time, W=band, C=dim -- so no permute is needed at the
   boundary at all (Torch's own ``forward`` does
   ``rearrange(x, "b t f c -> b c t f")`` to reach NCHW; skipping that
   rearrange here *is* the port). Every submodule below stays NHWC
   end-to-end. The one place this bites is ``_TFCTDFUnit``: Torch's TDF
   sub-block runs ``nn.Linear`` over the frequency axis, which is the last
   (contiguous) axis in NCHW for free, but is axis 2 in NHWC, so that block
   explicitly transposes width to the last axis around each ``Linear`` call
   and back for the surrounding channel-wise ``InstanceNorm``.
2. ``F.interpolate(..., mode="bilinear")`` is called 12 times, every one with
   an explicit ``size=`` and either an explicit or defaulted (None, which
   behaves as False here since no scale_factor is ever used)
   ``align_corners=False``. MLX's ``nn.Upsample`` only takes a
   ``scale_factor`` and derives the output size as ``int(scale * in)``, which
   can reproduce the wrong size when the true ratio isn't exact -- exactly
   the HyperACE branch-fusion case, where one branch is resized to another
   branch's unrelated H/W. ``_resize_bilinear`` below takes the target size
   directly and replicates PyTorch's half-pixel-center sampling formula
   instead; verified against ``F.interpolate`` on 6 shapes to <=3e-6 max abs
   error (float32 roundoff), see every call site's inline comment for the
   ``mode``/``align_corners`` it matches.

Reuses ``..model``'s ``MLP`` for the per-band mask path so that submodule
needs no new conversion rule beyond the ones the stock ``MaskEstimator``
already has. Every other submodule here is new and uses plain Python lists
(not ``mlx.nn.Sequential``) wherever Torch uses ``nn.Sequential``/
``nn.ModuleList``, so the resulting parameter paths are numerically indexed
exactly like Torch's (e.g. ``high_order_branch.0.cv1...``) with no
``.layers.`` insertion required -- see this module's docstring companion in
the porting report for the exact conv-weight-transpose and reshape rules
``convert.py`` still needs (not implemented here).

Reads: ..model.MLP, mlx.core, mlx.nn
"""

from __future__ import annotations

from typing import List, Tuple  # noqa: UP035

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from ..model import MLP


def _autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _resize_bilinear(x: mx.array, out_h: int, out_w: int) -> mx.array:
    """Resize an NHWC tensor's H/W to (out_h, out_w).

    Matches ``F.interpolate(x, size=(out_h, out_w), mode="bilinear",
    align_corners=False)`` -- see the module docstring for why this can't be
    ``mlx.nn.Upsample`` (no direct target-size API) and for the numeric
    verification against Torch.
    """
    b, h, w, c = x.shape  # noqa: RUF059
    if h == out_h and w == out_w:
        return x

    def _axis_gather(arr: mx.array, axis: int, n_in: int, n_out: int) -> mx.array:
        if n_in == n_out:
            return arr
        coords = (mx.arange(n_out, dtype=mx.float32) + 0.5) * (n_in / n_out) - 0.5
        coords = mx.clip(coords, 0, n_in - 1)
        idx_lo = mx.floor(coords)
        idx_hi = mx.minimum(idx_lo + 1, n_in - 1)
        frac = coords - idx_lo
        idx_lo = idx_lo.astype(mx.int32)
        idx_hi = idx_hi.astype(mx.int32)
        shape = [1] * arr.ndim
        shape[axis] = n_out
        frac = frac.reshape(shape)
        lo = mx.take(arr, idx_lo, axis=axis)
        hi = mx.take(arr, idx_hi, axis=axis)
        return lo * (1 - frac) + hi * frac

    x = _axis_gather(x, 1, h, out_h)
    x = _axis_gather(x, 2, w, out_w)
    return x


class ConvBlock(nn.Module):
    """Torch ``Conv``: conv(no bias) -> InstanceNorm(affine, eps=1e-8) -> SiLU/Identity."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=_autopad(k, p), groups=g, bias=False)
        self.bn = nn.InstanceNorm(c2, eps=1e-8, affine=True)
        self.act = nn.SiLU() if act else nn.Identity()

    def __call__(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConvBlock(nn.Module):
    """Torch ``DSConv``: depthwise conv -> pointwise conv -> InstanceNorm -> SiLU/Identity."""

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=_autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.InstanceNorm(c2, eps=1e-8, affine=True)
        self.act = nn.SiLU() if act else nn.Identity()

    def __call__(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConvBlock(c1, c_, k=3, s=1)
        self.dsconv2 = DSConvBlock(c_, c2, k=k, s=1)
        self.use_shortcut = shortcut and c1 == c2

    def __call__(self, x):
        out = self.dsconv2(self.dsconv1(x))
        return x + out if self.use_shortcut else out


class DSC3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c1, c_, 1, 1)
        self.cv3 = ConvBlock(2 * c_, c2, 1, 1)
        self.m = [DSBottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)]

    def __call__(self, x):
        y = self.cv1(x)
        for block in self.m:
            y = block(y)
        return self.cv3(mx.concatenate([y, self.cv2(x)], axis=-1))


class DSC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.m = DSC3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = ConvBlock(c_, c2, 1, 1)

    def __call__(self, x):
        return self.cv2(self.m(self.cv1(x)))


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads=8):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.global_proto = mx.zeros((num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x):
        # x: (b, n, c) -- n is the flattened spatial (h*w) axis.
        b, n, c = x.shape
        # F.adaptive_avg/max_pool1d(x.permute(0,2,1), 1).squeeze(-1) reduces to a
        # single value per (b, c) over n -- exactly a mean/max over axis=1 here.
        f_avg = mx.mean(x, axis=1)
        f_max = mx.max(x, axis=1)
        f_ctx = mx.concatenate([f_avg, f_max], axis=1)

        delta_p = mx.reshape(self.context_mapper(f_ctx), (b, self.num_hyperedges, c))
        proto = self.global_proto[None, :, :] + delta_p
        z = self.query_proj(x)
        z = mx.transpose(mx.reshape(z, (b, n, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        proto = mx.transpose(
            mx.reshape(proto, (b, self.num_hyperedges, self.num_heads, self.head_dim)), (0, 2, 3, 1)
        )
        sim = mx.matmul(z, proto) * self.scale
        s_bar = mx.mean(sim, axis=1)
        return mx.softmax(mx.transpose(s_bar, (0, 2, 1)), axis=-1)


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def __call__(self, x, a):
        f_m = mx.matmul(a, x)
        f_m = self.act(self.W_e(f_m))
        x_out = mx.matmul(mx.transpose(a, (0, 2, 1)), f_m)
        x_out = self.act(self.W_v(x_out))
        return x + x_out


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=8, num_heads=8):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads)
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def __call__(self, x):
        # x: NHWC (b, h, w, c). Torch flattens NCHW (b,c,h,w) -> (b,c,h*w) ->
        # permute -> (b,h*w,c); NHWC's own (h,w) axes are already channel-last
        # and contiguous in the same order, so a plain reshape is equivalent.
        b, h, w, c = x.shape
        x_flat = mx.reshape(x, (b, h * w, c))
        a = self.adaptive_hyperedge_gen(x_flat)
        x_out_flat = self.hypergraph_conv(x_flat, a)
        return mx.reshape(x_out_flat, (b, h, w, c))


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges=8, num_heads=8, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = ConvBlock(2 * c_, c2, 1, 1)

    def __call__(self, x):
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(mx.concatenate([x_ahc, x_lateral], axis=-1))


class HyperACE(nn.Module):
    def __init__(
        self,
        in_channels: List[int],  # noqa: UP006
        out_channels: int,
        num_hyperedges=8,
        num_heads=8,
        k=2,
        low_order_depth=1,
        c_h=0.5,
        c_l=0.25,
    ):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        c_mid = c4
        self.fuse_conv = ConvBlock(c2 + c3 + c4 + c5, c_mid, 1, 1)
        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        if self.c_s <= 0:
            raise ValueError("HyperACE channel split produced a non-positive skip channel count")

        self.high_order_branch = [C3AH(self.c_h, self.c_h, num_hyperedges, num_heads, e=1.0) for _ in range(k)]
        self.high_order_fuse = ConvBlock(self.c_h * k, self.c_h, 1, 1)
        self.low_order_branch = [DSC3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(low_order_depth)]
        self.final_fuse = ConvBlock(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def __call__(self, x: List[mx.array]) -> mx.array:  # noqa: UP006
        b2, b3, b4, b5 = x
        h4, w4 = b4.shape[1], b4.shape[2]
        # F.interpolate(..., mode="bilinear", align_corners=False), explicit in Torch.
        b2_resized = _resize_bilinear(b2, h4, w4)
        b3_resized = _resize_bilinear(b3, h4, w4)
        b5_resized = _resize_bilinear(b5, h4, w4)
        x_b = self.fuse_conv(mx.concatenate([b2_resized, b3_resized, b4, b5_resized], axis=-1))
        x_h, x_l, x_s = mx.split(x_b, [self.c_h, self.c_h + self.c_l], axis=-1)
        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(mx.concatenate(x_h_outs, axis=-1))
        x_l_out = x_l
        for block in self.low_order_branch:
            x_l_out = block(x_l_out)
        return self.final_fuse(mx.concatenate([x_h_fused, x_l_out, x_s], axis=-1))


class GatedFusion(nn.Module):
    """Torch ``GatedFusion``: ``f_in + gamma * h``, ``gamma`` shape (1, C, 1, 1).

    The Torch parameter is named ``gamma`` (converted generically to
    ``.weight`` by ``convert.py``'s existing ``.gamma`` -> ``.weight`` rule)
    and shaped ``(1, C, 1, 1)`` for NCHW broadcasting; stored here as a plain
    ``(C,)`` vector, which broadcasts the same way against NHWC's last axis
    and holds the identical per-channel values once the three singleton dims
    are squeezed out.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.weight = mx.zeros((in_channels,))

    def __call__(self, f_in, h):
        if f_in.shape[-1] != h.shape[-1]:
            raise ValueError(f"Channel mismatch: f_in={f_in.shape}, h={h.shape}")
        return f_in + self.weight * h


class Backbone(nn.Module):
    def __init__(self, in_channels=256, base_channels=64, base_depth=3):
        super().__init__()
        c2 = base_channels
        c3 = 256
        c4 = 384
        c5 = 512
        c6 = 768
        self.stem = DSConvBlock(in_channels, c2, k=3, s=(2, 1), p=1)
        self.p2 = [DSConvBlock(c2, c3, k=3, s=(2, 1), p=1), DSC3k2(c3, c3, n=base_depth)]
        self.p3 = [DSConvBlock(c3, c4, k=3, s=(2, 1), p=1), DSC3k2(c4, c4, n=base_depth * 2)]
        self.p4 = [DSConvBlock(c4, c5, k=3, s=2, p=1), DSC3k2(c5, c5, n=base_depth * 2)]
        self.p5 = [DSConvBlock(c5, c6, k=3, s=2, p=1), DSC3k2(c6, c6, n=base_depth)]
        self.out_channels = [c3, c4, c5, c6]

    def __call__(self, x):
        x = self.stem(x)
        x2 = self.p2[1](self.p2[0](x))
        x3 = self.p3[1](self.p3[0](x2))
        x4 = self.p4[1](self.p4[0](x3))
        x5 = self.p5[1](self.p5[0](x4))
        return [x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, encoder_channels: List[int], hyperace_out_c: int, decoder_channels: List[int]):  # noqa: UP006
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d2, c_d3, c_d4, c_d5 = decoder_channels
        self.h_to_d5 = ConvBlock(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = ConvBlock(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = ConvBlock(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = ConvBlock(hyperace_out_c, c_d2, 1, 1)
        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)
        self.skip_p5 = ConvBlock(c_p5, c_d5, 1, 1)
        self.skip_p4 = ConvBlock(c_p4, c_d4, 1, 1)
        self.skip_p3 = ConvBlock(c_p3, c_d3, 1, 1)
        self.skip_p2 = ConvBlock(c_p2, c_d2, 1, 1)
        self.up_d5 = DSC3k2(c_d5, c_d4, n=1)
        self.up_d4 = DSC3k2(c_d4, c_d3, n=1)
        self.up_d3 = DSC3k2(c_d3, c_d2, n=1)
        self.final_d2 = DSC3k2(c_d2, c_d2, n=1)

    def __call__(self, enc_feats: List[mx.array], h_ace: mx.array):  # noqa: UP006
        p2, p3, p4, p5 = enc_feats
        d5 = self.skip_p5(p5)
        # Every F.interpolate below: mode="bilinear", align_corners left at Torch's
        # default (None) -- which behaves as False here since size= is always given
        # explicitly and no scale_factor path is ever taken.
        h_d5 = self.h_to_d5(_resize_bilinear(h_ace, d5.shape[1], d5.shape[2]))
        d5 = self.fusion_d5(d5, h_d5)
        d5_up = _resize_bilinear(d5, p4.shape[1], p4.shape[2])
        d4 = self.up_d5(d5_up) + self.skip_p4(p4)
        h_d4 = self.h_to_d4(_resize_bilinear(h_ace, d4.shape[1], d4.shape[2]))
        d4 = self.fusion_d4(d4, h_d4)
        d4_up = _resize_bilinear(d4, p3.shape[1], p3.shape[2])
        d3 = self.up_d4(d4_up) + self.skip_p3(p3)
        h_d3 = self.h_to_d3(_resize_bilinear(h_ace, d3.shape[1], d3.shape[2]))
        d3 = self.fusion_d3(d3, h_d3)
        d3_up = _resize_bilinear(d3, p2.shape[1], p2.shape[2])
        d2 = self.up_d3(d3_up) + self.skip_p2(p2)
        h_d2 = self.h_to_d2(_resize_bilinear(h_ace, d2.shape[1], d2.shape[2]))
        d2 = self.fusion_d2(d2, h_d2)
        return self.final_d2(d2)


class _TFCTDFUnit(nn.Module):
    """One element of Torch ``TFCTDF.blocks`` (there, an ad hoc ``nn.Module()``
    with ``tfc1``/``tdf``/``tfc2``/``shortcut`` attached directly).

    ``tfc1``/``tfc2`` are channel-wise (InstanceNorm/SiLU/Conv2d, all NHWC-native).
    ``tdf`` is the layout hazard: Torch's ``nn.Linear`` inside it runs over the
    frequency axis, which is contiguous last-axis in NCHW for free but is axis 2
    (not the channel axis) in NHWC here -- so width is explicitly swapped to the
    last axis around each ``Linear`` call and swapped back for the surrounding
    channel-wise ``InstanceNorm``.
    """

    def __init__(self, in_c, c, f, bn=4):
        super().__init__()
        self.tfc1 = [
            nn.InstanceNorm(in_c, eps=1e-8, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_c, c, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        self.tdf = [
            nn.InstanceNorm(c, eps=1e-8, affine=True),
            nn.SiLU(),
            nn.Linear(f, f // bn, bias=False),
            nn.InstanceNorm(c, eps=1e-8, affine=True),
            nn.SiLU(),
            nn.Linear(f // bn, f, bias=False),
        ]
        self.tfc2 = [
            nn.InstanceNorm(c, eps=1e-8, affine=True),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        self.shortcut = nn.Conv2d(in_c, c, kernel_size=1, stride=1, padding=0, bias=False)

    def _run_channelwise(self, layers, x):
        for layer in layers:
            x = layer(x)
        return x

    def _run_tdf(self, x):
        layers = self.tdf
        x = layers[0](x)  # InstanceNorm, channel-last: fine as-is
        x = layers[1](x)  # SiLU
        x = mx.transpose(x, (0, 1, 3, 2))  # (b,h,w,c) -> (b,h,c,w): expose freq axis last
        x = layers[2](x)  # Linear(f, f // bn) over the (now-last) frequency axis
        x = mx.transpose(x, (0, 1, 3, 2))  # back to (b,h,w',c)
        x = layers[3](x)  # InstanceNorm
        x = layers[4](x)  # SiLU
        x = mx.transpose(x, (0, 1, 3, 2))
        x = layers[5](x)  # Linear(f // bn, f)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x

    def __call__(self, x):
        s = self.shortcut(x)
        x = self._run_channelwise(self.tfc1, x)
        x = x + self._run_tdf(x)
        x = self._run_channelwise(self.tfc2, x)
        return x + s


class TFCTDF(nn.Module):
    def __init__(self, in_c, c, depth, f, bn=4):
        super().__init__()
        self.blocks = []
        for _ in range(depth):
            self.blocks.append(_TFCTDFUnit(in_c, c, f, bn=bn))
            in_c = c

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FreqPixelShuffle(nn.Module):
    """Torch views channels as (out_c, scale), permutes, then folds ``scale``
    into width -- a channel-to-width pixel shuffle. Reproduced directly in NHWC
    (see the class body): reshape the last (channel) axis into (out_c, scale),
    swap ``scale`` next to width, then merge (width, scale) -- no NCHW tensor is
    ever formed.
    """

    def __init__(self, in_channels, out_channels, scale, f):
        super().__init__()
        self.scale = scale
        self.conv = DSConvBlock(in_channels, out_channels * scale)
        self.out_conv = TFCTDF(out_channels, out_channels, 2, f)

    def __call__(self, x):
        x = self.conv(x)
        b, h, w, c_r = x.shape
        out_c = c_r // self.scale
        x = mx.reshape(x, (b, h, w, out_c, self.scale))
        x = mx.transpose(x, (0, 1, 2, 4, 3))
        x = mx.reshape(x, (b, h, w * self.scale, out_c))
        return self.out_conv(x)


class ProgressiveUpsampleHead(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025, in_bands=62):
        super().__init__()
        self.target_bins = target_bins
        c = in_channels
        self.block1 = FreqPixelShuffle(c, c // 2, scale=2, f=in_bands * 2)
        self.block2 = FreqPixelShuffle(c // 2, c // 4, scale=2, f=in_bands * 4)
        self.block3 = FreqPixelShuffle(c // 4, c // 8, scale=2, f=in_bands * 8)
        self.block4 = FreqPixelShuffle(c // 8, c // 16, scale=2, f=in_bands * 16)
        # Torch: kernel_size=3, stride=1, padding="same" -- "same" for k=3/s=1/dilation=1
        # resolves to padding=1 exactly; MLX's Conv2d has no string-padding mode.
        self.final_conv = nn.Conv2d(c // 16, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if x.shape[2] != self.target_bins:
            # F.interpolate(..., mode="bilinear", align_corners=False), explicit in Torch.
            x = _resize_bilinear(x, x.shape[1], self.target_bins)
        return self.final_conv(x)


class SegmModel(nn.Module):
    def __init__(
        self,
        in_bands=62,
        in_dim=256,
        out_bins=1025,
        out_channels=4,
        base_channels=64,
        base_depth=2,
        num_hyperedges=32,
        num_heads=8,
    ):
        super().__init__()
        self.backbone = Backbone(in_channels=in_dim, base_channels=base_channels, base_depth=base_depth)
        enc_channels = self.backbone.out_channels
        _, _, c4, _ = enc_channels
        hyperace_out_channels = c4
        self.hyperace = HyperACE(
            enc_channels,
            hyperace_out_channels,
            num_hyperedges,
            num_heads,
            k=2,
            low_order_depth=1,
        )
        decoder_channels = list(enc_channels)
        self.decoder = Decoder(enc_channels, hyperace_out_channels, decoder_channels)
        self.upsample_head = ProgressiveUpsampleHead(
            in_channels=decoder_channels[0],
            out_channels=out_channels,
            target_bins=out_bins,
            in_bands=in_bands,
        )

    def __call__(self, x):
        h = x.shape[1]  # NHWC: time is axis 1
        enc_feats = self.backbone(x)
        h_ace_feats = self.hyperace(enc_feats)
        dec_feat = self.decoder(enc_feats, h_ace_feats)
        # F.interpolate(..., size=(h, dec_feat.shape[-1]), mode="bilinear",
        # align_corners=False), explicit in Torch: restores time, leaves width alone.
        feat_time_restored = _resize_bilinear(dec_feat, h, dec_feat.shape[2])
        return self.upsample_head(feat_time_restored)


class HyperACEMaskEstimator(nn.Module):
    """MLX port of pcunwa's HyperACE v2 mask-estimator head.

    Signature matches the Torch constructor call site
    (``bs_roformer.py``'s ``_create_mask_estimator`` / the MLX
    ``heads.build_mask_estimator`` registry) exactly: ``dim``, ``dim_inputs``,
    ``depth`` (per-band mask MLP depth), ``audio_channels``,
    ``mlp_expansion_factor``.
    """

    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],  # noqa: UP006
        depth,
        audio_channels=2,
        mlp_expansion_factor=4,
    ):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.num_bands = len(dim_inputs)
        dim_hidden = dim * mlp_expansion_factor

        self._mlp_module_names: list[str] = []
        for i, dim_in in enumerate(dim_inputs):
            module_name = f"to_freqs_{i}"
            setattr(self, module_name, MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth))
            self._mlp_module_names.append(module_name)

        out_channels = 2 * audio_channels
        self.segm = SegmModel(
            in_bands=len(dim_inputs),
            in_dim=dim,
            out_bins=sum(dim_inputs) // out_channels,
            out_channels=out_channels,
        )

    def __call__(self, x):
        # x: (b, t, f, dim) is already NHWC (H=time, W=band, C=dim) -- no permute.
        y = self.segm(x)
        b, t, f, c = y.shape
        y = mx.reshape(y, (b, t, f * c))

        outs = []
        for i in range(self.num_bands):
            mlp = getattr(self, self._mlp_module_names[i])
            band_features = x[..., i, :]
            freq_out_before_glu = mlp(band_features)
            values, gates = mx.split(freq_out_before_glu, 2, axis=-1)
            outs.append(values * mx.sigmoid(gates))

        return mx.concatenate(outs, axis=-1) + y


__all__ = ["HyperACEMaskEstimator"]
