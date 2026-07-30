"""PyTorch backend -- the shipped path, unchanged, behind the seam.

Holds an already-constructed model plus its resolved device and turns a mixture
into stems by calling the same utils.demix_track() the package has always used;
it is a wrapper, never a second implementation, so the Torch path cannot drift
from the advanced composition API that callers still import directly.

The per-track ETA carried between tracks lives here because it is a property of
Torch's chunk timing, not of folder iteration.

Reads: ..utils (demix_track), .base (SeparationBackend), torch, numpy
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from ..utils import demix_track


class TorchBackend:
    """Wraps a loaded BSRoformer and its device as a SeparationBackend."""

    name = "torch"

    def __init__(self, model, config, device):
        self._model = model.eval()
        self._config = config
        self._device = device
        self._first_chunk_time = None

    @classmethod
    def is_available(cls) -> bool:
        return True

    @property
    def resolved_device(self) -> str:
        return str(self._device)

    @property
    def model(self):
        """The resident model, for callers composing the advanced path directly."""
        return self._model

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        if self._first_chunk_time is not None:
            self._print_estimate(mix.shape[1])
        mixture = torch.tensor(mix, dtype=torch.float32)
        result, self._first_chunk_time = demix_track(
            self._config, self._model, mixture, self._device, self._first_chunk_time
        )
        return result

    def release(self) -> None:
        from ..inference import mps_available

        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if mps_available():
            torch.mps.empty_cache()

    def _print_estimate(self, total_length: int) -> None:
        step = self._config.inference.chunk_size // self._config.inference.num_overlap
        num_chunks = (total_length + step - 1) // step
        estimated = self._first_chunk_time * num_chunks
        print(f"Estimated total processing time for this track: {estimated:.2f} seconds")
        sys.stdout.write(f"Estimated time remaining: {estimated:.2f} seconds\r")
        sys.stdout.flush()
