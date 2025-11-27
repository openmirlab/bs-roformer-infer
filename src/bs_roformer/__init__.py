"""BS-Roformer inference utilities."""

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

__version__ = "0.1.0"
