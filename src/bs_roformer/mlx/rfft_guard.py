"""`exact_zero_safe_rfft` -- routes `mx.fft.rfft` through the CPU stream.

Not model architecture: this is a workaround for an MLX 0.31.2 Metal kernel bug,
kept in its own module (mirroring the same-named guard in the sibling
`bs-mamba2-infer` package) so it reads as infrastructure rather than getting lost
inside the trunk model file it protects. `model.py`'s `__call__` and
`heads/fno.py`'s `_SpectralConv1D` both wrap their `mx.fft.rfft` calls in this
context manager for the reason explained in its docstring below -- read that
before touching either call site or this file.

Reads: mlx.core
"""

from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def exact_zero_safe_rfft():
    """Route `mx.fft.rfft` through the CPU stream for one STFT. NOT upstream code.

    MLX 0.31.2's Metal rfft kernel packs two real FFTs into one complex FFT; in
    float32 that cancellation is not bit-exact, so a frame whose true value is
    exactly zero comes back as roughly 4.5e-07 instead of 0. That matters far more
    than its size suggests: `L2Norm` discards magnitude, and its eps of 1e-12 is
    five orders below the artifact, so the clamp never engages and pure numerical
    noise is normalized into a full-scale, essentially random feature vector --
    about a millionfold amplification. Time-axis attention then spreads that one
    corrupted frame across every position, which is why silence at the end of a
    chunk corrupts the output at the beginning.

    This is not a corner case: every track's final chunk is padded, and music has
    rests. Measured on the real checkpoint, a zero-padded chunk diverged from Torch
    by 1.455e-02 max abs; with this workaround, 2.012e-07 -- the same noise floor
    as a chunk with no silence at all.

    Raising `L2Norm`'s eps was considered and rejected: genuinely quiet audio has
    legitimate band norms in the same range, so it would trade this bug for a
    quieter one that only shows up on soft material.

    Caveat, stated rather than hidden: this swaps a module-level attribute, so it
    is not thread-safe. Inference here is single-threaded per session.

    Delete this once MLX's kernel is fixed. See brain/evidence.md section 14.
    """
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original
