"""Public package surface -- re-exports the model, registry, and CLI entry points.

Thin barrel: no logic of its own, just the stable import path (`from bs_roformer
import ...`) that callers, tests, and the `bs-roformer-infer` / `bs-roformer-download`
console scripts in pyproject.toml depend on.

Reads: .bs_roformer, .utils, .inference, .download, .model_registry
"""

from .bs_roformer import BSRoformer
from .utils import get_model_from_config, demix_track
from .inference import main as inference_main
from .download import main as download_main
from .model_registry import MODEL_REGISTRY, BSModel, DEFAULT_MODEL

__all__ = [
    "BSRoformer",
    "BSModel",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "get_model_from_config",
    "demix_track",
    "inference_main",
    "download_main",
]

__version__ = "0.1.2"
