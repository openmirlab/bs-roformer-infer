"""Transition parity: legacy JSON fixtures must exactly match runtime TOML."""
from __future__ import annotations

import json
from pathlib import Path

from bs_roformer import download
from bs_roformer.checkpoints import artifact_metadata, checkpoint_metadata
from bs_roformer.model_registry import MODEL_REGISTRY


DATA = Path(__file__).parents[1] / "src" / "bs_roformer" / "data"
CKPT_BASE = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
CONFIG_BASE = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"


def test_toml_matches_every_legacy_model_artifact_and_download_lookup():
    models = json.loads((DATA / "bs_models.json").read_text())["models"]
    checksums = json.loads((DATA / "checksums.json").read_text())["files"]
    overrides = json.loads((DATA / "overrides.json").read_text())
    assert set(models) == {model.slug for model in MODEL_REGISTRY.list()}
    for slug, legacy in models.items():
        runtime = checkpoint_metadata(slug)
        assert runtime["name"] == legacy["name"]
        assert runtime["category"] == legacy["category"]
        model = MODEL_REGISTRY.get(slug)
        for kind, filename, override_map, base in (
            ("checkpoint", legacy["checkpoint"], overrides["checkpoints"], CKPT_BASE),
            ("config", legacy["config"], overrides["configs"], CONFIG_BASE),
        ):
            artifact = artifact_metadata(slug, kind)
            expected_url = override_map.get(filename, base + filename)
            assert artifact["name"] == filename
            assert artifact["url"] == expected_url
            assert artifact["sha256"] == checksums[filename]["sha256"]
            assert artifact["size"] == checksums[filename]["size"]
            assert download._expected_checksum(filename) == (
                checksums[filename]["sha256"], checksums[filename]["size"]
            )
            resolved_url = download._checkpoint_url(model) if kind == "checkpoint" else download._config_url(model)
            if kind == "config" and (Path(download.PACKAGE_ROOT) / "configs" / filename).exists():
                assert resolved_url is None
            else:
                assert resolved_url == expected_url
