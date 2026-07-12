#!/usr/bin/env python3
"""Download BS-Roformer checkpoints/configs -- CLI, sha256 verification, models-dir resolution.

Third-party HF accounts can vanish without warning: the jarredou account behind the
original BS-RoFormer-SW checkpoint was deleted (discovered 2026-06), 404ing the
default model's download URL for every user until 557f791 repointed it. data/overrides.json
is the patch point for exactly this failure mode -- when a URL starts 404ing, edit
that file first, before touching this module's code. Resolution precedence per asset:
overrides.json entry > packaged local file under configs/ (configs only, checked by
_copy_packaged_config) > DEFAULT_CKPT_BASE_URL / DEFAULT_CONFIG_BASE_URL construction.

Downloads are sha256-verified against data/checksums.json (mismatch = delete + retry,
never a silently kept corrupt file); assets without a recorded hash fall back to the
non-empty-size check and say so. The models directory resolves as: explicit argument
> $BS_ROFORMER_MODELS_PATH > ~/.cache/bs-roformer-infer, with a legacy ./models
directory honored as a read fallback so pre-0.1.4 users don't re-download.
ensure_model_assets() is the auto-download entry point inference.py uses on first run.

Reads: .model_registry (BSModel, MODEL_REGISTRY, DEFAULT_MODEL), data/overrides.json,
data/checksums.json, requests, tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import requests
from tqdm import tqdm

from .model_registry import DEFAULT_MODEL, MODEL_REGISTRY, BSModel

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data"
CACHE_DIR_NAME = "bs-roformer-infer"
MODELS_DIR_ENV = "BS_ROFORMER_MODELS_PATH"
LEGACY_MODELS_DIR = Path("models")
DEFAULT_CKPT_BASE_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
DEFAULT_CONFIG_BASE_URL = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"

STATIC_CHECKPOINT_OVERRIDES = {}


def _load_override_maps():
    path = DATA_ROOT / "overrides.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    return data.get("checkpoints", {}), data.get("configs", {})


_CHECKPOINT_OVERRIDES_EXTRA, _CONFIG_OVERRIDES_EXTRA = _load_override_maps()
CHECKPOINT_URL_OVERRIDES = {**STATIC_CHECKPOINT_OVERRIDES, **_CHECKPOINT_OVERRIDES_EXTRA}
CONFIG_URL_OVERRIDES = _CONFIG_OVERRIDES_EXTRA


def _load_checksums() -> dict:
    path = DATA_ROOT / "checksums.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("files", {})


CHECKSUMS = _load_checksums()


def default_models_dir() -> Path:
    """Directory new downloads go to: $BS_ROFORMER_MODELS_PATH > ~/.cache/bs-roformer-infer."""
    env = os.environ.get(MODELS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / CACHE_DIR_NAME


def models_search_dirs(models_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Where existing assets are looked for, in order.

    First the download target (explicit argument > env var > cache default), then
    the legacy relative ./models directory pre-0.1.4 releases downloaded into, so
    existing users keep their files without re-downloading.
    """
    dirs = [Path(models_dir).expanduser() if models_dir else default_models_dir()]
    legacy = LEGACY_MODELS_DIR
    if legacy.is_dir() and legacy.resolve() != dirs[0].resolve():
        dirs.append(legacy)
    return dirs


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def verify_file_integrity(file_path: Path, expected_size: Optional[int] = None,
                          expected_sha256: Optional[str] = None) -> bool:
    """Verify file integrity: existence, non-emptiness, optional size and sha256 match."""
    if not file_path.exists():
        return False

    # Check file size if expected size is provided
    if expected_size and file_path.stat().st_size != expected_size:
        print(f"Warning: File size mismatch. Expected: {expected_size}, Got: {file_path.stat().st_size}")
        return False

    # Basic validation - file should not be empty
    if file_path.stat().st_size == 0:
        print(f"Error: Downloaded file is empty: {file_path}")
        return False

    # Strongest check: sha256 against the recorded known-good hash
    if expected_sha256:
        actual = get_file_hash(file_path)
        if actual != expected_sha256:
            print(f"Error: SHA256 mismatch for {file_path}")
            print(f"  expected: {expected_sha256}")
            print(f"  actual:   {actual}")
            return False

    return True


def download_file(url: str, output_path: Path, description: str = "Downloading",
                 expected_size: Optional[int] = None, max_retries: int = 3,
                 expected_sha256: Optional[str] = None) -> bool:
    """
    Download a file with progress bar, sha256/size verification, and retry logic.

    Args:
        url: URL to download from
        output_path: Local path to save the file
        description: Description for progress bar
        expected_size: Expected file size in bytes (optional)
        max_retries: Maximum number of retry attempts
        expected_sha256: Known-good sha256 hex digest to verify against (optional)

    Returns:
        True if download successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {description} (attempt {attempt + 1}/{max_retries})...")
            
            # Use requests for Hugging Face downloads
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify download integrity
            if verify_file_integrity(output_path, expected_size, expected_sha256):
                print(f"✓ Successfully downloaded: {output_path}")
                print(f"  File size: {output_path.stat().st_size:,} bytes")
                if expected_sha256:
                    print(f"  SHA256 verified: {expected_sha256}")
                return True
            else:
                print(f"✗ Download verification failed: {output_path}")
                if output_path.exists():
                    output_path.unlink()  # Remove corrupted file
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return False
                
        except Exception as e:
            print(f"✗ Error downloading {description} (attempt {attempt + 1}): {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Remove partial file
            
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Failed to download {description} after {max_retries} attempts")
                return False
    
    return False


def _dedupe(models: Iterable[BSModel]) -> List[BSModel]:
    seen = set()
    ordered = []
    for model in models:
        if model.slug in seen:
            continue
        seen.add(model.slug)
        ordered.append(model)
    return ordered


def _resolve_models(args) -> List[BSModel]:
    if args.models:
        selected = []
        for key in args.models:
            try:
                selected.append(MODEL_REGISTRY.get(key))
            except KeyError:
                print(f"❌ Unknown model identifier: {key}")
                continue
        return _dedupe(selected)

    if args.categories:
        selected = []
        for category in args.categories:
            entries = MODEL_REGISTRY.list(category.lower())
            if not entries:
                print(f"⚠️ No models found for category '{category}'")
            selected.extend(entries)
        return _dedupe(selected)

    if args.all:
        return MODEL_REGISTRY.list()

    # default to BS-Roformer SW checkpoint
    try:
        return [MODEL_REGISTRY.get("roformer-model-bs-roformer-sw-by-jarredou")]
    except KeyError:
        return []


def _checkpoint_url(model: BSModel) -> Optional[str]:
    return CHECKPOINT_URL_OVERRIDES.get(
        model.checkpoint,
        f"{DEFAULT_CKPT_BASE_URL}{model.checkpoint}"
    )


def _config_url(model: BSModel) -> Optional[str]:
    packaged = PACKAGE_ROOT / "configs" / model.config
    if packaged.exists():
        return None
    if model.config in CONFIG_URL_OVERRIDES:
        return CONFIG_URL_OVERRIDES[model.config]
    return f"{DEFAULT_CONFIG_BASE_URL}{model.config}"


def _copy_packaged_config(model: BSModel, target_path: Path) -> bool:
    source = PACKAGE_ROOT / "configs" / model.config
    if not source.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    print(f"✓ Copied packaged config -> {target_path}")
    return True


def _expected_checksum(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """Recorded (sha256, size) for a downloadable asset, or (None, None) if unrecorded."""
    entry = CHECKSUMS.get(filename)
    if not entry:
        return None, None
    return entry.get("sha256"), entry.get("size")


def _download_checkpoint(model: BSModel, model_dir: Path, force: bool) -> bool:
    target = model_dir / model.checkpoint
    if target.exists() and not force:
        print(f"✓ {model.checkpoint} already exists, skipping checkpoint download")
        return True

    url = _checkpoint_url(model)
    if not url:
        print(f"❌ No download URL known for {model.name}")
        return False
    sha256, size = _expected_checksum(model.checkpoint)
    if sha256 is None:
        print(f"⚠️ No recorded sha256 for {model.checkpoint}; only basic size checks will run")
    print(f"\n🔄 Downloading {model.name} checkpoint...")
    return download_file(url, target, f"{model.name} checkpoint",
                         expected_size=size, expected_sha256=sha256)


def _download_config(model: BSModel, model_dir: Path, force: bool) -> bool:
    target = model_dir / model.config
    if target.exists() and not force:
        print(f"✓ {model.config} already exists, skipping config download")
        return True
    if _copy_packaged_config(model, target):
        return True
    url = _config_url(model)
    if not url:
        print(f"❌ No config source known for {model.name}")
        return False
    sha256, size = _expected_checksum(model.config)
    if sha256 is None:
        print(f"⚠️ No recorded sha256 for {model.config}; only basic size checks will run")
    print(f"\n🔄 Downloading {model.name} config...")
    return download_file(url, target, f"{model.name} config",
                         expected_size=size, expected_sha256=sha256)


def download_model_assets(models: Sequence[BSModel],
                          output_dir: Path,
                          *,
                          models_only: bool = False,
                          config_only: bool = False,
                          force: bool = False) -> bool:
    success = True
    for model in models:
        model_dir = output_dir / model.slug
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📦 Preparing {model.name} ({model.category})")
        if not config_only:
            success &= _download_checkpoint(model, model_dir, force)
        if not models_only:
            success &= _download_config(model, model_dir, force)
    return success


def ensure_model_assets(model: Union[str, BSModel] = DEFAULT_MODEL,
                        models_dir: Optional[Union[str, Path]] = None,
                        *,
                        download_missing: bool = True) -> Tuple[Path, Path]:
    """Resolve (checkpoint_path, config_path) for a registry model, downloading if missing.

    Searches models_search_dirs() for an existing checkpoint first (so explicitly
    configured dirs, the env var dir, the cache default, and legacy ./models all
    keep working); only if none has it does the sha256-verified download run into
    the first search dir. Raises on failure instead of returning half a model.
    """
    entry = MODEL_REGISTRY.get(model) if isinstance(model, str) else model
    search_dirs = models_search_dirs(models_dir)

    for base in search_dirs:
        model_dir = base / entry.slug
        checkpoint = model_dir / entry.checkpoint
        if checkpoint.exists():
            config = model_dir / entry.config
            if not config.exists() and not _download_config(entry, model_dir, force=False):
                raise RuntimeError(
                    f"Found checkpoint for {entry.name} in {model_dir} but could not "
                    f"obtain its config ({entry.config})"
                )
            return checkpoint, model_dir / entry.config

    if not download_missing:
        raise FileNotFoundError(
            f"No local copy of {entry.name} ({entry.checkpoint}) in: "
            + ", ".join(str(d) for d in search_dirs)
        )

    model_dir = search_dirs[0] / entry.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {entry.name} not found locally; downloading to {model_dir}")
    if not _download_checkpoint(entry, model_dir, force=False):
        raise RuntimeError(f"Failed to download checkpoint for {entry.name} -- see log above")
    if not _download_config(entry, model_dir, force=False):
        raise RuntimeError(f"Failed to obtain config for {entry.name} -- see log above")
    return model_dir / entry.checkpoint, model_dir / entry.config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BS-Roformer model weights and configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bs-roformer-download --list-models
  bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou
  bs-roformer-download --category vocals --output-dir ./models
  bs-roformer-download --all --models-only
        """
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to store downloaded assets "
            f"(default: ${MODELS_DIR_ENV} if set, else ~/.cache/{CACHE_DIR_NAME})"
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model slug, checkpoint filename, or friendly name (repeatable)"
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Model category filter (karaoke, vocals, instrumental, denoise, ...). Repeatable."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List registry entries and exit"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every BS-Roformer checkpoint from the registry"
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Download checkpoints only"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Download configs only"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.models_only and args.config_only:
        print("❌ Error: Cannot specify both --models-only and --config-only")
        sys.exit(1)

    if args.list_models:
        print(MODEL_REGISTRY.as_table())
        print("\nAvailable categories:", ", ".join(MODEL_REGISTRY.categories()))
        return

    selected_models = _resolve_models(args)
    if not selected_models:
        print("No models selected. Use --list-models to see available options.")
        return

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_models_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    success = download_model_assets(
        selected_models,
        output_dir,
        models_only=args.models_only,
        config_only=args.config_only,
        force=args.force,
    )

    print("\n" + "=" * 50)
    if success:
        print("🎉 Downloads complete")
        print(f"📁 Assets stored in: {output_dir}")
    else:
        print("❌ Some downloads failed. Check the logs and try again with --force if needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
