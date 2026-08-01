#!/usr/bin/env python3
"""Strict-load and short-forward probe for every direct pcunwa BS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ml_collections import ConfigDict

from bs_roformer import MODEL_REGISTRY, ensure_model_assets
from bs_roformer.checkpoints import load_checkpoints
from bs_roformer.inference import SafeLoaderWithTuple
from bs_roformer.utils import get_model_from_config, load_checkpoint_state


def pcunwa_slugs():
    data = load_checkpoints()["models"]
    return [
        slug
        for slug, model in data.items()
        if any(
            artifact["kind"] == "checkpoint"
            and "huggingface.co/pcunwa/" in artifact["url"]
            for artifact in model["artifacts"]
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    slugs = args.models or pcunwa_slugs()

    for slug in slugs:
        entry = MODEL_REGISTRY.get(slug)
        checkpoint, config_path = ensure_model_assets(entry, models_dir=args.models_dir)
        with config_path.open() as handle:
            config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
        model = get_model_from_config(
            "bs_roformer", config, model_variation=entry.variation
        )
        model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
        model.eval()
        if not args.skip_forward:
            channels = 2 if model.stereo else 1
            with torch.no_grad():
                output = model(torch.zeros(1, channels, args.samples))
            if not torch.isfinite(output).all():
                raise RuntimeError(f"non-finite forward output for {slug}")
        print(f"OK {slug} ({entry.variation})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
