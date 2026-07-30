"""The backend seam -- one narrow protocol every compute backend implements.

A backend owns everything framework-specific: model construction, checkpoint
weights, tensor layout, device placement, and the chunked overlap-add itself.
Chunking accumulates on-device for speed, so the seam deliberately sits at a whole
mixture rather than a single chunk -- a per-chunk seam would drag every
accumulator back to the host and hand the acceleration straight back.

Above the seam nothing knows a dtype, a tensor layout, or which chip is busy:
file iteration, stem naming, instrumental derivation, and the output manifest are
decided once in inference.separate_folder_with().

Reads: numpy (boundary array type only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_CHUNK_SIZE = 588800


@dataclass(frozen=True)
class ChunkingPlan:
    """How one mixture is cut into overlapping chunks -- one owner, both backends.

    The framework-specific loops differ, but these numbers must not: two backends
    deriving `step` or `fade_size` independently is the same design decision
    encoded twice, and they would drift silently rather than loudly.
    """

    chunk_size: int
    num_overlap: int
    step: int
    fade_size: int
    border: int

    @classmethod
    def from_config(cls, config) -> ChunkingPlan:
        # chunk_size moved between config sections across upstream config versions.
        if hasattr(config.inference, "chunk_size"):
            chunk_size = config.inference.chunk_size
        elif hasattr(config, "audio") and hasattr(config.audio, "chunk_size"):
            chunk_size = config.audio.chunk_size
        else:
            chunk_size = DEFAULT_CHUNK_SIZE
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
        return cls(
            chunk_size=chunk_size,
            num_overlap=num_overlap,
            step=step,
            fade_size=chunk_size // 10,
            border=chunk_size - step,
        )


@runtime_checkable
class SeparationBackend(Protocol):
    """Turns one mixture into stems, hiding how and where it computed them."""

    #: Stable identifier, matching the `backend=` argument that selects it.
    name: str

    @classmethod
    def is_available(cls) -> bool:
        """True when this backend can actually run on this machine right now."""

    @property
    def resolved_device(self) -> str:
        """The concrete target chosen, after any `auto` sentinel was resolved."""

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        """Separate one `(channels, samples)` float32 mixture into named stems.

        Every returned array has the same shape as `mix`.
        """

    def release(self) -> None:
        """Drop resident model and device memory. Disk checkpoints stay."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend is requested by name but cannot run here.

    Always raised, never swallowed into a fallback: silently substituting a
    different backend discards what the caller explicitly asked for, and they
    would only discover it by noticing the wrong hardware was busy.
    """
