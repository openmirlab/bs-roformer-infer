"""Packaged-config and override-resolution regression tests for the download pipeline.

Covers two release-blocking bugs: (1) the packaged BS-Roformer-SW config was named
differently from BSModel.config, so `_copy_packaged_config` never matched and the
yaml was silently network-fetched every time -- fixed by renaming the bundled file
to BS-Rofo-SW-Fixed.yaml (2026-07); (2) the jarredou HF account outage (2026-06),
where a future silent revert of overrides.json to a dead URL should fail these tests
instead of shipping quietly.

Reads: bs_roformer.download, bs_roformer.model_registry.MODEL_REGISTRY
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bs_roformer import download as download_module
from bs_roformer.model_registry import MODEL_REGISTRY, DEFAULT_MODEL

SW_SLUG = "roformer-model-bs-roformer-sw-by-jarredou"
ENERJAZZER_CKPT_URL = (
    "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt"
)
ENERJAZZER_CONFIG_URL = (
    "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml"
)


class TestPackagedConfigFix:
    """The SW model's config should resolve to the bundled file, not a network fetch."""

    def test_default_model_is_sw(self):
        assert DEFAULT_MODEL == SW_SLUG

    def test_packaged_config_file_matches_model_config_name(self):
        model = MODEL_REGISTRY.get(SW_SLUG)
        packaged = download_module.PACKAGE_ROOT / "configs" / model.config
        assert packaged.exists(), (
            f"Bundled config filename must match BSModel.config ({model.config!r}) "
            f"or _copy_packaged_config will never find it and the config always "
            f"gets network-fetched instead."
        )

    def test_config_url_is_none_when_packaged_copy_exists(self):
        """_config_url returns None (meaning: use the local file) once the names line up."""
        model = MODEL_REGISTRY.get(SW_SLUG)
        assert download_module._config_url(model) is None

    def test_download_config_uses_packaged_copy_without_network(self, tmp_path, monkeypatch):
        """_download_config must hit the packaged-config path, never download_file."""
        model = MODEL_REGISTRY.get(SW_SLUG)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "download_file() was called -- packaged config was not used"
            )

        monkeypatch.setattr(download_module, "download_file", _fail_if_called)

        model_dir = tmp_path / model.slug
        model_dir.mkdir(parents=True)
        ok = download_module._download_config(model, model_dir, force=True)

        assert ok is True
        copied = model_dir / model.config
        assert copied.exists()
        assert copied.read_bytes() == (
            download_module.PACKAGE_ROOT / "configs" / model.config
        ).read_bytes()


class TestOverrideResolutionOutageRegression:
    """Guards against a silent revert of overrides.json to the dead jarredou URLs."""

    def test_checkpoint_override_present_for_sw_model(self):
        model = MODEL_REGISTRY.get(SW_SLUG)
        assert model.checkpoint in download_module.CHECKPOINT_URL_OVERRIDES, (
            f"No checkpoint override registered for {model.checkpoint!r}; "
            f"it would fall back to the default UVR base URL, which does not host this file."
        )

    def test_checkpoint_override_points_at_enerjazzer_mirror(self):
        model = MODEL_REGISTRY.get(SW_SLUG)
        url = download_module._checkpoint_url(model)
        assert url == ENERJAZZER_CKPT_URL
        assert "enerjazzer" in url
        assert "jarredou" not in url

    def test_config_override_points_at_enerjazzer_mirror(self):
        """Even though the packaged copy short-circuits this, the override entry itself
        must stay correct as a fallback for anyone who deletes their local install's
        packaged config or runs against an older sdist."""
        overrides = download_module.CONFIG_URL_OVERRIDES
        model = MODEL_REGISTRY.get(SW_SLUG)
        assert model.config in overrides, (
            f"No config override registered for {model.config!r}."
        )
        url = overrides[model.config]
        assert url == ENERJAZZER_CONFIG_URL
        assert "enerjazzer" in url
        assert "jarredou" not in url


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
