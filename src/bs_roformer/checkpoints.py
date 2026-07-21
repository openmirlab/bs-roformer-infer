"""Package-owned checkpoint metadata and validation."""
from pathlib import Path
import hashlib
import tomllib

def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"

def load_checkpoints(path=None) -> dict:
    path = Path(path) if path else checkpoint_config_path()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc
    if data.get("schema", {}).get("version") != 1 or not isinstance(data.get("models"), dict):
        raise ValueError("checkpoint config must define schema.version=1 and models")
    for key, model in data["models"].items():
        if not isinstance(model, dict) or not isinstance(model.get("artifacts"), list):
            raise ValueError(f"invalid checkpoint metadata for {key}")
        for artifact in model["artifacts"]:
            if not isinstance(artifact, dict) or not str(artifact.get("url", "")).startswith("https://"):
                raise ValueError(f"invalid artifact URL for {key}")
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ValueError(f"invalid artifact SHA-256 for {key}")
    return data

def checkpoint_metadata(model: str) -> dict:
    data = load_checkpoints()
    try:
        return dict(data["models"][model])
    except KeyError as exc:
        raise KeyError(f"unknown checkpoint model: {model}") from exc

def artifact_metadata(model: str, kind: str) -> dict:
    for artifact in checkpoint_metadata(model)["artifacts"]:
        if artifact.get("kind") == kind:
            return dict(artifact)
    raise KeyError(f"missing {kind} artifact for {model}")

__all__ = ["checkpoint_config_path", "load_checkpoints", "checkpoint_metadata", "artifact_metadata"]
