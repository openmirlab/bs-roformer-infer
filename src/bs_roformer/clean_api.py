"""Explicit lifecycle facade for BS-Roformer inference and output manifests.

This module keeps the public Python API small: session lifecycle, lazy loading,
and a thin inference call that delegates to inference.run_folder(). The session
surfaces the exact output files run_folder() wrote, instead of inventing metadata
from the registry, so callers can treat it like a receipt for the work just done.

Reads: .checkpoints (checkpoint_metadata), .download (ensure_model_assets,
get_file_hash), .inference (OutputManifest, SafeLoaderWithTuple, run_folder),
.utils (get_model_from_config), yaml, ml_collections, torch
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ml_collections import ConfigDict

from .checkpoints import checkpoint_metadata


class BSRoformerSession:
    def __init__(
        self,
        *,
        model_name="roformer-model-bs-roformer-sw-by-jarredou",
        model=None,
        model_path=None,
        config_path=None,
        config=None,
        models_dir=None,
        device=None,
        progress=True,
        checkpoint_url=None,
        checkpoint_sha256=None,
    ):
        self.model_name = model_name
        self._model = model
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self.models_dir = models_dir
        self.device = device
        self.progress = progress
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self._config = config
        self._status = "ready" if model is not None else "new"

    @property
    def status(self):
        return self._status

    def _metadata(self):
        try:
            return checkpoint_metadata(self.model_name)
        except KeyError:
            return {"artifacts": []}

    def load(self):
        if self._status == "ready":
            return self
        if self._status == "closed":
            raise RuntimeError("cannot load a closed BSRoformerSession")
        self._status = "loading"
        try:
            from .download import ensure_model_assets, get_file_hash
            from .inference import SafeLoaderWithTuple
            from .utils import get_model_from_config

            if self.model_path is None or self.config_path is None:
                self.model_path, self.config_path = ensure_model_assets(
                    self.model_name,
                    models_dir=self.models_dir,
                )
            with self.config_path.open() as handle:
                self._config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
            self._model = get_model_from_config("bs_roformer", self._config)

            import torch

            artifacts = self._metadata().get("artifacts", [])
            expected = self.checkpoint_sha256 or next(
                (
                    artifact["sha256"]
                    for artifact in artifacts
                    if artifact.get("kind") == "checkpoint"
                ),
                None,
            )
            if expected and get_file_hash(self.model_path).lower() != expected.lower():
                raise ValueError(f"checkpoint SHA-256 mismatch for {self.model_path}")
            state = torch.load(self.model_path, map_location="cpu")
            self._model.load_state_dict(state)
            from .inference import _select_device
            from argparse import Namespace
            target = _select_device(Namespace(device=self.device))
            self._model = self._model.to(target).eval()
            self.device = target
            self._status = "ready"
            return self
        except Exception:
            self._status = "failed"
            raise

    def infer(self, input_folder, *, store_dir="outputs", verbose=False):
        if self._status != "ready" or self._model is None:
            raise RuntimeError("BSRoformerSession must be ready; call load() before infer()")
        from .inference import run_folder
        from argparse import Namespace

        return run_folder(
            self._model,
            Namespace(input_folder=Path(input_folder), store_dir=Path(store_dir)),
            self._config,
            self.device,
            verbose=verbose,
        )

    def release(self):
        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if self._status != "closed":
            self._status = "released"

    def close(self):
        self.release()
        self._status = "closed"

    def cache_info(self):
        resolved_checkpoint = self.model_path
        if resolved_checkpoint is None:
            from .download import ensure_model_assets
            try:
                resolved_checkpoint, _ = ensure_model_assets(
                    self.model_name, models_dir=self.models_dir, download_missing=False
                )
            except FileNotFoundError:
                pass
        return {
            "model": self.model_name,
            "status": self._status,
            "model_loaded": self._model is not None,
            "models_dir": str(self.models_dir) if self.models_dir else None,
            "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint else None,
            "checkpoint_url": self.checkpoint_url
            or next(
                (
                    artifact["url"]
                    for artifact in self._metadata().get("artifacts", [])
                    if artifact.get("kind") == "checkpoint"
                ),
                None,
            ),
        }

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class BSRoformerSeparator:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = BSRoformerSession(**self._kwargs).load()
        return self._session

    def __call__(self, input_folder, *, store_dir="outputs", verbose=False):
        return self.session.infer(input_folder, store_dir=store_dir, verbose=verbose)


def separate_folder(input_folder, **kwargs):
    allowed = {
        "model_name",
        "model_path",
        "config_path",
        "models_dir",
        "device",
        "progress",
        "checkpoint_url",
        "checkpoint_sha256",
    }
    session_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return BSRoformerSeparator(**session_kwargs).session.infer(
        input_folder,
        store_dir=kwargs.get("store_dir", "outputs"),
    )
