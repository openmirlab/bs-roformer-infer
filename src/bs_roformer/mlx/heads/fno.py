"""FNOMaskEstimator -- MLX port of the FNO mask-estimator head.

Mirrors ``bs_roformer.fno.FNOMaskEstimator``: a per-band FNO1d (3 spectral-conv
blocks with pointwise-conv/soft-gating skips) followed by a channel-axis GLU,
matching the minimal inference-only FNO1d surface the Torch original
reimplements rather than depending on ``neuraloperator``. Every ``Conv1d`` in
the Torch source has kernel size 1, so this port never reaches for
``mlx.nn.Conv1d`` (which would force an NCL<->NLC transpose fight with the
FFT, which wants the time axis last) -- a kernel-size-1 conv is just a
per-position linear map over channels, done here with one ``mx.einsum`` in
``_Conv1x1`` while keeping the channels-first ``(b, c, t)`` layout Torch uses
throughout. The one correctness-critical piece is ``_SpectralConv1D``, which
truncates/multiplies/pads in the rfft domain like Torch's ``SpectralConv1D``
and wraps its ``mx.fft.rfft`` call in ``exact_zero_safe_rfft()`` (imported
from ``..model``): MLX 0.31.2's Metal rfft kernel returns roughly 4.5e-07
instead of exact 0 for an all-zero band, and the un-normalized fp32 arithmetic
through the two skip-summed branches here would otherwise carry that artifact
into the mask. ``depth``/``mlp_expansion_factor`` are accepted only to match
the shared mask-estimator constructor signature -- the Torch class never uses
either (FNO1d layer count is hard-coded to 3 and modes to 64 in Torch, and
this mirrors that exactly). See ``convert.py`` for the extra weight-name
mapping rules this head's module tree requires (the two existing
``mask_estimators.`` / ``to_freqs.N.0.`` rules already do the outer
restructuring; the FNO-internal submodule lists still need their own rules).

Reads: mlx.core, mlx.nn, ..model (ExactGELU, exact_zero_safe_rfft)
"""

import mlx.core as mx
import mlx.nn as nn  # noqa: PLR0402

from ..model import ExactGELU, exact_zero_safe_rfft


class _Conv1x1(nn.Module):
    """Kernel-size-1 Conv1d equivalent on channels-first ``(b, c, t)`` arrays.

    Weight is kept at Torch's native Conv1d shape ``(out, in, 1)`` (rather
    than squeezed to ``(out, in)``) so a converted checkpoint's tensor can be
    copied in verbatim with no value-level reshape, only a key rename.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, 1))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x):
        w = self.weight[:, :, 0]
        y = mx.einsum("bit,oi->bot", x, w)
        bias = getattr(self, "bias", None)
        if bias is not None:
            y = y + bias.reshape(1, -1, 1)
        return y


class _ChannelMLP(nn.Module):
    """Stack of ``_Conv1x1`` layers with exact GELU between (not after the last)."""

    def __init__(self, in_channels, out_channels, hidden_channels=None, n_layers=2):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.n_layers = n_layers
        for index in range(n_layers):
            if index == 0 and index == n_layers - 1:
                layer_in, layer_out = in_channels, out_channels
            elif index == 0:
                layer_in, layer_out = in_channels, hidden_channels
            elif index == n_layers - 1:
                layer_in, layer_out = hidden_channels, out_channels
            else:
                layer_in, layer_out = hidden_channels, hidden_channels
            setattr(self, f"fcs_{index}", _Conv1x1(layer_in, layer_out))
        self._act = ExactGELU()

    def __call__(self, x):
        for index in range(self.n_layers):
            x = getattr(self, f"fcs_{index}")(x)
            if index < self.n_layers - 1:
                x = self._act(x)
        return x


class _SoftGating(nn.Module):
    """Learned per-channel scale, broadcasting ``(1, c, 1)`` over ``(b, c, t)``."""

    def __init__(self, channels):
        super().__init__()
        self.weight = mx.ones((1, channels, 1))

    def __call__(self, x):
        return self.weight * x


class _SpectralConv1D(nn.Module):
    """Separable FNO1d spectral conv: one complex weight per (channel, mode).

    Truncates to ``min(fft_width, num_modes)`` frequency bins (matching
    Torch's slice-in/zero-pad-out ``SpectralConv1D.forward``), multiplies by
    the learned per-channel-per-mode complex weight, and zero-pads back out to
    the full rfft width before the inverse transform. ``fft_norm="forward"``
    in Torch maps directly to ``norm="forward"`` on both ``mx.fft.rfft`` and
    ``mx.fft.irfft``.
    """

    def __init__(self, channels, n_modes_height):
        super().__init__()
        self.num_modes = n_modes_height // 2 + 1
        self.weight = mx.zeros((channels, self.num_modes), dtype=mx.complex64)
        self.bias = mx.zeros((channels, 1))

    def __call__(self, x):
        b, c, width = x.shape
        fft_width = width // 2 + 1
        with exact_zero_safe_rfft():
            x_ft = mx.fft.rfft(x, axis=-1, norm="forward")
        modes = min(fft_width, self.num_modes)
        weight_active = self.weight[:, :modes]
        x_active = x_ft[:, :, :modes] * weight_active[None, :, :]
        pad_amount = fft_width - modes
        if pad_amount > 0:
            out_ft = mx.concatenate(
                [x_active, mx.zeros((b, c, pad_amount), dtype=x_active.dtype)],
                axis=-1,
            )
        else:
            out_ft = x_active
        x_out = mx.fft.irfft(out_ft, n=width, axis=-1, norm="forward")
        return x_out + self.bias


class _FNOBlocks(nn.Module):
    """``n_layers`` spectral-conv blocks, each with a pointwise-conv skip and
    a channel-MLP-with-soft-gating-skip, GELU between blocks (not after the last)."""

    def __init__(self, channels, n_modes_height, n_layers):
        super().__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"convs_{i}", _SpectralConv1D(channels, n_modes_height))
            setattr(self, f"fno_skips_{i}", _Conv1x1(channels, channels, bias=False))
            setattr(
                self,
                f"channel_mlp_{i}",
                _ChannelMLP(channels, channels, hidden_channels=round(channels * 0.5), n_layers=2),
            )
            setattr(self, f"channel_mlp_skips_{i}", _SoftGating(channels))
        self._act = ExactGELU()

    def __call__(self, x, index):
        fno_skip = getattr(self, f"fno_skips_{index}")(x)
        mlp_skip = getattr(self, f"channel_mlp_skips_{index}")(x)
        x = getattr(self, f"convs_{index}")(x) + fno_skip
        if index < self.n_layers - 1:
            x = self._act(x)
        x = getattr(self, f"channel_mlp_{index}")(x) + mlp_skip
        if index < self.n_layers - 1:
            x = self._act(x)
        return x


class _FNO1D(nn.Module):
    """Grid-embed -> lift -> ``n_layers`` FNO blocks -> project, channels-first."""

    def __init__(self, *, n_modes_height, hidden_channels, in_channels, out_channels, n_layers):
        super().__init__()
        lifting_channels = 2 * hidden_channels
        projection_channels = 2 * hidden_channels
        self.lifting = _ChannelMLP(in_channels + 1, hidden_channels, hidden_channels=lifting_channels, n_layers=2)
        self.fno_blocks = _FNOBlocks(hidden_channels, n_modes_height, n_layers)
        self.projection = _ChannelMLP(hidden_channels, out_channels, hidden_channels=projection_channels, n_layers=2)
        self.n_layers = n_layers

    def __call__(self, x):
        b, _c, t = x.shape
        grid = mx.linspace(0.0, 1.0, t + 1)[:-1]
        grid = mx.broadcast_to(mx.reshape(grid, (1, 1, t)), (b, 1, t))
        x = mx.concatenate([x, grid], axis=1)
        x = self.lifting(x)
        for index in range(self.n_layers):
            x = self.fno_blocks(x, index)
        return self.projection(x)


class FNOMaskEstimator(nn.Module):
    """MLX port of pcunwa's FNO mask-estimator head.

    Signature matches the Torch constructor call site
    (``bs_roformer.py``'s ``_create_mask_estimator``) exactly: ``dim``,
    ``dim_inputs``, ``depth``, ``mlp_expansion_factor``. Torch's
    ``FNOMaskEstimator`` never reads ``depth`` or ``mlp_expansion_factor``
    (each per-band FNO1d is hard-coded to ``n_modes_height=64``,
    ``n_layers=3``) -- both are accepted here only to keep the constructor
    interchangeable with the other mask-estimator heads.
    """

    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.num_bands = len(self.dim_inputs)
        for i, dim_in in enumerate(self.dim_inputs):
            setattr(
                self,
                f"to_freqs_{i}",
                _FNO1D(
                    n_modes_height=64,
                    hidden_channels=dim,
                    in_channels=dim,
                    out_channels=dim_in * 2,
                    n_layers=3,
                ),
            )

    def __call__(self, x):
        outs = []
        for i in range(self.num_bands):
            band_features = x[..., i, :]  # (b, t, dim)
            band_features = mx.transpose(band_features, (0, 2, 1))  # (b, dim, t)
            freq_out = getattr(self, f"to_freqs_{i}")(band_features)  # (b, dim_in * 2, t)
            values, gates = mx.split(freq_out, 2, axis=1)
            band_out = values * mx.sigmoid(gates)  # (b, dim_in, t)
            outs.append(mx.transpose(band_out, (0, 2, 1)))  # (b, t, dim_in)
        return mx.concatenate(outs, axis=-1)


__all__ = ["FNOMaskEstimator"]
