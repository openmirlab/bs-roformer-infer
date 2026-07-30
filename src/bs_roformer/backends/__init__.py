"""Backend registry -- the only name callers use to pick a compute path.

Resolution is by name and nothing more: asking for a backend that cannot run here
raises rather than quietly substituting another, because a silent substitution is
discovered by noticing the wrong hardware was busy. Backend modules are imported
lazily so `import bs_roformer` never drags an optional framework into the default
import path.

Reads: .base (SeparationBackend, BackendUnavailable), .torch_backend (lazily)
"""

from __future__ import annotations

from .base import BackendUnavailable, SeparationBackend

#: Every selectable backend name, in the order `auto` prefers them.
BACKEND_NAMES = ("mlx", "torch")
DEFAULT_BACKEND = "torch"


def _load(name: str):
    """Import a backend module on demand. Importing must not require its framework."""
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend
    raise BackendUnavailable(f"no backend registered under {name!r}")


def resolve_backend_name(requested: str | None, *, variation: str | None = None) -> str:
    """Resolve a requested backend name, honouring it exactly or raising.

    `None` and `"torch"` both mean the shipped Torch path -- the default never
    moves on its own. `"auto"` prefers an accelerated backend when one is
    genuinely importable *and* can run this checkpoint, falling back to Torch
    otherwise; that is the one place a fallback is what the caller asked for.

    `variation` is the checkpoint's registry mask-estimator variation, when known.
    Passing it matters: without it, `auto` can settle on a backend that has no head
    for the model and then fail at construction, where Torch would simply have
    worked. An *explicit* backend still raises for a variation it cannot run --
    that request is honoured or refused, never downgraded.
    """
    if requested is None or requested == DEFAULT_BACKEND:
        return DEFAULT_BACKEND
    if requested == "auto":
        for name in BACKEND_NAMES:
            try:
                backend = _load(name)
            except BackendUnavailable:
                continue
            if backend.is_available() and _supports(backend, variation):
                return name
        return DEFAULT_BACKEND
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"backend must be None, 'auto', or one of {BACKEND_NAMES}; got {requested!r}"
        )
    backend = _load(requested)
    if not backend.is_available():
        raise BackendUnavailable(
            f"backend {requested!r} is unavailable on this machine; "
            f"it may need an optional extra (pip install 'bs-roformer-infer[{requested}]')"
        )
    if variation is not None and not _supports(backend, variation):
        backend.assert_supports_variation(variation)
    return requested


def _supports(backend, variation) -> bool:
    """Whether `backend` can run a checkpoint carrying this registry variation."""
    check = getattr(backend, "supports_variation", None)
    return True if check is None else bool(check(variation))


def get_backend(name: str):
    """Return the backend class registered under `name`."""
    return _load(name)


__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_BACKEND",
    "BackendUnavailable",
    "SeparationBackend",
    "get_backend",
    "resolve_backend_name",
]
