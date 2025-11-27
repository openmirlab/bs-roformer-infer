#!/usr/bin/env python3
"""Download script for BS-Roformer checkpoints/configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests
from tqdm import tqdm

from .model_registry import MODEL_REGISTRY, BSModel

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data"
DEFAULT_OUTPUT = Path("models")
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


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def verify_file_integrity(file_path: Path, expected_size: Optional[int] = None) -> bool:
    """Verify file integrity by checking size and basic validation."""
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
    
    return True


def download_file(url: str, output_path: Path, description: str = "Downloading", 
                 expected_size: Optional[int] = None, max_retries: int = 3) -> bool:
    """
    Download a file with progress bar and retry logic.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        description: Description for progress bar
        expected_size: Expected file size in bytes (optional)
        max_retries: Maximum number of retry attempts
    
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
            if verify_file_integrity(output_path, expected_size):
                print(f"✓ Successfully downloaded: {output_path}")
                print(f"  File size: {output_path.stat().st_size:,} bytes")
                return True
            else:
                print(f"✗ Download verification failed: {output_path}")
                if output_path.exists():
                    output_path.unlink()  # Remove corrupted file
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return False
                
        except Exception as e:
            print(f"✗ Error downloading {description} (attempt {attempt + 1}): {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Remove partial file
            
            if attempt < max_retries - 1:
                print(f"Retrying in 2 seconds...")
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


def _download_checkpoint(model: BSModel, model_dir: Path, force: bool) -> bool:
    target = model_dir / model.checkpoint
    if target.exists() and not force:
        print(f"✓ {model.checkpoint} already exists, skipping checkpoint download")
        return True

    url = _checkpoint_url(model)
    if not url:
        print(f"❌ No download URL known for {model.name}")
        return False
    print(f"\n🔄 Downloading {model.name} checkpoint...")
    return download_file(url, target, f"{model.name} checkpoint")


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
    print(f"\n🔄 Downloading {model.name} config...")
    return download_file(url, target, f"{model.name} config")


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
        default=str(DEFAULT_OUTPUT),
        help="Directory to store downloaded assets (default: models)"
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

    output_dir = Path(args.output_dir).expanduser()
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
