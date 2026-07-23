"""FNO mask-estimator variation for BS-RoFormer checkpoints.

The pcunwa FNO checkpoint replaces the original per-band MLP MaskEstimator with
``neuraloperator``'s FNO1d. Pulling that full research framework into this
inference package would add training/logging/storage dependencies that the
checkpoint does not need, so this module implements only the FNO1d inference
surface whose state_dict keys appear in the published checkpoint.

Reads: torch, einops
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from beartype import beartype
from einops import rearrange


class DenseTensor(Module):
    """Small state_dict-compatible wrapper for neuraloperator's dense tensor weight."""

    def __init__(self, shape):
        super().__init__()
        self.tensor = nn.Parameter(torch.empty(shape, dtype=torch.cfloat))

    def __getitem__(self, indices):
        return self.tensor[tuple(indices)] if isinstance(indices, list) else self.tensor[indices]


class GridEmbedding1D(Module):
    def __init__(self):
        super().__init__()
        self._grid = None
        self._resolution = None

    def forward(self, x):
        resolution = x.shape[-1:]
        if self._grid is None or self._resolution != resolution:
            grid = torch.linspace(0.0, 1.0, resolution[0] + 1, device=x.device, dtype=x.dtype)[:-1]
            self._grid = grid.view(1, 1, resolution[0])
            self._resolution = resolution
        return torch.cat((x, self._grid.expand(x.shape[0], -1, -1)), dim=1)


class ChannelMLP(Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_layers=2):
        super().__init__()
        self.fcs = ModuleList([])
        hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        for index in range(n_layers):
            if index == 0 and index == n_layers - 1:
                self.fcs.append(nn.Conv1d(in_channels, out_channels, 1))
            elif index == 0:
                self.fcs.append(nn.Conv1d(in_channels, hidden_channels, 1))
            elif index == n_layers - 1:
                self.fcs.append(nn.Conv1d(hidden_channels, out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(hidden_channels, hidden_channels, 1))

    def forward(self, x):
        for index, layer in enumerate(self.fcs):
            x = layer(x)
            if index < len(self.fcs) - 1:
                x = F.gelu(x)
        return x


class Flattened1dConv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        size = list(x.shape)
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        return x.view(size[0], self.conv.out_channels, *size[2:])


class SoftGating(Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return self.weight * x


class SpectralConv1D(Module):
    def __init__(
        self,
        channels,
        n_modes,
        *,
        separable=True,
        fft_norm="forward",
    ):
        super().__init__()
        if not separable:
            raise ValueError("Only separable FNO1d weights are supported by this inference variation")
        self.in_channels = channels
        self.out_channels = channels
        self.separable = separable
        self.fft_norm = fft_norm
        self.n_modes = [n_modes // 2 + 1]
        self.max_n_modes = list(self.n_modes)
        self.weight = DenseTensor((channels, *self.max_n_modes))
        init_std = (2 / (channels + channels)) ** 0.5
        self.bias = nn.Parameter(init_std * torch.randn(channels, 1))

    def transform(self, x, output_shape=None):
        if output_shape is not None and tuple(x.shape[2:]) != tuple(output_shape):
            return F.interpolate(x, size=output_shape, mode="linear", align_corners=True)
        return x

    def forward(self, x, output_shape=None):
        batch_size, _, width = x.shape
        fft_width = width // 2 + 1
        x_ft = torch.fft.rfftn(x, norm=self.fft_norm, dim=(-1,))
        out_ft = torch.zeros(
            [batch_size, self.out_channels, fft_width],
            device=x.device,
            dtype=torch.cfloat,
        )

        start = self.max_n_modes[0] - min(fft_width, self.n_modes[0])
        weight_slice = [slice(None), slice(None, -start) if start else slice(None)]
        weight = self.weight[weight_slice]
        x_start = fft_width - min(fft_width, weight.shape[1])
        x_slice = [slice(None), slice(None), slice(None, -x_start) if x_start else slice(None)]
        x_slice = tuple(x_slice)
        out_ft[x_slice] = x_ft[x_slice] * weight

        if output_shape is not None:
            width = output_shape[0]
        x = torch.fft.irfftn(out_ft, s=(width,), dim=(-1,), norm=self.fft_norm)
        return x + self.bias


class FNOBlocks(Module):
    def __init__(self, channels, n_modes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.convs = ModuleList(
            [SpectralConv1D(channels, n_modes, separable=True) for _ in range(n_layers)]
        )
        self.fno_skips = ModuleList([Flattened1dConv(channels, channels) for _ in range(n_layers)])
        self.channel_mlp = ModuleList(
            [
                ChannelMLP(channels, channels, hidden_channels=round(channels * 0.5), n_layers=2)
                for _ in range(n_layers)
            ]
        )
        self.channel_mlp_skips = ModuleList([SoftGating(channels) for _ in range(n_layers)])

    def forward(self, x, index=0, output_shape=None):
        fno_skip = self.convs[index].transform(self.fno_skips[index](x), output_shape=output_shape)
        mlp_skip = self.convs[index].transform(
            self.channel_mlp_skips[index](x),
            output_shape=output_shape,
        )
        x = self.convs[index](x, output_shape=output_shape) + fno_skip
        if index < self.n_layers - 1:
            x = F.gelu(x)
        x = self.channel_mlp[index](x) + mlp_skip
        if index < self.n_layers - 1:
            x = F.gelu(x)
        return x


class FNO1D(Module):
    def __init__(
        self,
        *,
        n_modes_height,
        hidden_channels,
        in_channels,
        out_channels,
        lifting_channels,
        projection_channels,
        n_layers,
        separable=True,
    ):
        super().__init__()
        if not separable:
            raise ValueError("Only separable FNO1d is supported by this inference variation")
        self.positional_embedding = GridEmbedding1D()
        lifting_channels = 2 * hidden_channels
        projection_channels = 2 * hidden_channels
        self.lifting = ChannelMLP(
            in_channels + 1,
            hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
        )
        self.fno_blocks = FNOBlocks(hidden_channels, n_modes_height, n_layers)
        self.projection = ChannelMLP(
            hidden_channels,
            out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
        )

    def forward(self, x):
        x = self.positional_embedding(x)
        x = self.lifting(x)
        for index in range(self.fno_blocks.n_layers):
            x = self.fno_blocks(x, index)
        return self.projection(x)


class FNOMaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        mlp_expansion_factor=4,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        for dim_in in dim_inputs:
            self.to_freqs.append(
                nn.Sequential(
                    FNO1D(
                        n_modes_height=64,
                        hidden_channels=dim,
                        in_channels=dim,
                        out_channels=dim_in * 2,
                        lifting_channels=dim,
                        projection_channels=dim,
                        n_layers=3,
                        separable=True,
                    ),
                    nn.GLU(dim=-2),
                )
            )

    def forward(self, x):
        outs = []
        for band_features, fno in zip(x.unbind(dim=-2), self.to_freqs):
            band_features = rearrange(band_features, "b t c -> b c t")
            with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32):
                freq_out = fno(band_features).float()
            outs.append(rearrange(freq_out, "b c t -> b t c"))
        return torch.cat(outs, dim=-1)


__all__ = ["FNOMaskEstimator"]
