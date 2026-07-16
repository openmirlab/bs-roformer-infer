"""Public package surface -- re-exports the model, registry, and CLI entry points.

Thin barrel: no logic of its own, just the stable import path (`from bs_roformer
import ...`) that callers, tests, and the `bs-roformer-infer` / `bs-roformer-download`
console scripts in pyproject.toml depend on.

Reads: .__about__, .bs_roformer, .utils, .inference, .download, .model_registry
"""

from .__about__ import __version__
from .bs_roformer import BSRoformer
from .utils import get_model_from_config, demix_track
from .inference import OutputFile, OutputManifest, main as inference_main
from .download import (
    default_models_dir,
    ensure_model_assets,
    main as download_main,
)
from .model_registry import MODEL_REGISTRY, BSModel, DEFAULT_MODEL
from .clean_api import BSRoformerSession, BSRoformerSeparator, separate_folder

__all__ = [
    "BSRoformer",
    "BSModel",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "get_model_from_config",
    "demix_track",
    "OutputFile",
    "OutputManifest",
    "inference_main",
    "download_main",
    "ensure_model_assets",
    "default_models_dir",
    "__version__",
    "BSRoformerSession",
    "BSRoformerSeparator",
    "separate_folder",
]
