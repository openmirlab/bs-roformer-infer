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
        backend=None,
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
        self.backend = backend
        self._backend = None
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
            from .backends import get_backend, resolve_backend_name

            # Resolve the backend before any expensive work: an unavailable one
            # must fail here, not after a 700MB checkpoint has been verified. The
            # registry variation is read first so `auto` can skip a backend that
            # has no head for this checkpoint instead of failing at construction.
            self.backend = resolve_backend_name(
                self.backend, variation=self._metadata().get("variation")
            )

            from .download import ensure_model_assets
            from .inference import SafeLoaderWithTuple

            if self.model_path is None or self.config_path is None:
                self.model_path, self.config_path = ensure_model_assets(
                    self.model_name,
                    models_dir=self.models_dir,
                )
            with self.config_path.open() as handle:
                self._config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))

            if self.backend == "torch":
                self._load_torch()
            else:
                self._verify_checkpoint_hash()
                self._backend = get_backend(self.backend).from_checkpoint(
                    config=self._config,
                    checkpoint_path=self.model_path,
                    variation=self._metadata().get("variation"),
                )
                self._model = self._backend.model
                self.device = self._backend.resolved_device
            self._status = "ready"
            return self
        except Exception:
            self._status = "failed"
            raise

    def _verify_checkpoint_hash(self):
        """Refuse a checkpoint whose bytes do not match the recorded digest."""
        from .download import get_file_hash

        expected = self.checkpoint_sha256 or next(
            (
                artifact["sha256"]
                for artifact in self._metadata().get("artifacts", [])
                if artifact.get("kind") == "checkpoint"
            ),
            None,
        )
        if expected and get_file_hash(self.model_path).lower() != expected.lower():
            raise ValueError(f"checkpoint SHA-256 mismatch for {self.model_path}")

    def _load_torch(self):
        """The shipped Torch construction path, preserved step for step."""
        from argparse import Namespace

        from .backends import get_backend
        from .inference import _select_device
        from .utils import get_model_from_config, load_checkpoint_state

        self._model = get_model_from_config(
            "bs_roformer",
            self._config,
            model_variation=self._metadata().get("variation"),
        )
        self._verify_checkpoint_hash()
        self._model.load_state_dict(load_checkpoint_state(self.model_path, map_location="cpu"))
        target = _select_device(Namespace(device=self.device))
        self._model = self._model.to(target).eval()
        self.device = target
        self._backend = get_backend("torch")(self._model, self._config, target)

    def infer(self, input_folder, *, store_dir="outputs", verbose=False, output_format="wav_float32"):
        if self._status != "ready" or self._model is None:
            raise RuntimeError("BSRoformerSession must be ready; call load() before infer()")
        from argparse import Namespace

        from .inference import separate_folder_with

        return separate_folder_with(
            self._ensure_backend().separate,
            Namespace(input_folder=Path(input_folder), store_dir=Path(store_dir)),
            self._config,
            verbose=verbose,
            output_format=output_format,
        )

    def _ensure_backend(self):
        """The backend, built on demand for sessions handed a model directly."""
        if self._backend is None:
            from .backends import get_backend, resolve_backend_name

            self.backend = resolve_backend_name(self.backend)
            self._backend = get_backend(self.backend)(self._model, self._config, self.device)
        return self._backend

    def release(self):
        # Defer entirely to the backend when there is one: it already moves the
        # model off-device and clears the right cache. Doing it here as well
        # called .cpu() twice on the same model.
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        elif self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        try:
            import torch

            from .inference import mps_available

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if mps_available():
                torch.mps.empty_cache()
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
            "backend": self.backend,
            "device": str(self.device) if self.device is not None else None,
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

    def __call__(self, input_folder, *, store_dir="outputs", verbose=False, output_format="wav_float32"):
        return self.session.infer(
            input_folder, store_dir=store_dir, verbose=verbose, output_format=output_format
        )


def separate_folder(input_folder, **kwargs):
    allowed = {
        "model_name",
        "model_path",
        "config_path",
        "models_dir",
        "device",
        "backend",
        "progress",
        "checkpoint_url",
        "checkpoint_sha256",
    }
    session_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return BSRoformerSeparator(**session_kwargs).session.infer(
        input_folder,
        store_dir=kwargs.get("store_dir", "outputs"),
        output_format=kwargs.get("output_format", "wav_float32"),
    )
