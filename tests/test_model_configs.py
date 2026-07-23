"""Config-loading + model-instantiation regression tests for every bundled/downloaded config.

Guards three things that have broken before: (1) every config loads with
yaml.safe_load() with no !!python/tuple errors, (2) the list-to-tuple conversion in
get_model_from_config actually runs, (3) models instantiate without beartype
errors. Also runnable standalone (`python tests/test_model_configs.py`) for a
human-readable pass/fail summary outside pytest.

Reads: bs_roformer.utils.get_model_from_config, ml_collections, yaml
"""

import sys
from pathlib import Path

import pytest
import yaml
from ml_collections import ConfigDict

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bs_roformer.utils import get_model_from_config, load_checkpoint_state


def find_all_configs() -> list[tuple[str, Path]]:
    """Find all config files in the project."""
    project_root = Path(__file__).parent.parent
    configs = []

    # Bundled configs in src
    bundled_dir = project_root / "src" / "bs_roformer" / "configs"
    for cfg in bundled_dir.glob("*.yaml"):
        configs.append((f"bundled:{cfg.name}", cfg))

    # Downloaded model configs in models/
    models_dir = project_root / "models"
    if models_dir.exists():
        for cfg in models_dir.glob("**/*.yaml"):
            configs.append((f"models:{cfg.parent.name}/{cfg.name}", cfg))

    return configs


def load_config_safe(config_path: Path) -> ConfigDict:
    """Load config with yaml.safe_load (as inference.py does)."""
    with open(config_path) as f:
        return ConfigDict(yaml.safe_load(f))


class TestConfigLoading:
    """Test that configs can be loaded with safe_load."""

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_config_loads_with_safe_load(self, name: str, config_path: Path):
        """Config should load without !!python/tuple errors."""
        config = load_config_safe(config_path)
        assert config is not None
        assert "model" in config

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_multi_stft_param_is_list(self, name: str, config_path: Path):
        """After safe_load, multi_stft_resolutions_window_sizes should be a list."""
        config = load_config_safe(config_path)
        param = config.model.get("multi_stft_resolutions_window_sizes")
        if param is not None:
            assert isinstance(param, list), (
                f"Expected list from yaml.safe_load, got {type(param).__name__}"
            )


class TestModelInstantiation:
    """Test that models can be instantiated with the list-to-tuple fix."""

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_model_instantiates(self, name: str, config_path: Path):
        """Model should instantiate without beartype errors."""
        config = load_config_safe(config_path)

        # Skip if missing required inference params (incomplete config)
        if not hasattr(config, "training"):
            pytest.skip(f"Config missing 'training' section: {name}")

        model = get_model_from_config("bs_roformer", config)
        assert model is not None

        # Verify the model has the expected attribute
        assert hasattr(model, "multi_stft_resolutions_window_sizes")
        assert isinstance(model.multi_stft_resolutions_window_sizes, tuple), (
            "Model param should be tuple after conversion"
        )

    def test_mlp_expansion_factor_reaches_mask_estimator(self):
        """Downloaded configs can vary the mask-estimator MLP width."""
        config = ConfigDict(
            {
                "model": {
                    "dim": 8,
                    "depth": 1,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "freqs_per_bands": [512, 513],
                    "dim_head": 4,
                    "heads": 1,
                    "dim_freqs_in": 1025,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 512,
                    "stft_win_length": 2048,
                    "mask_estimator_depth": 2,
                    "mlp_expansion_factor": 2,
                },
                "training": {"target_instrument": "vocals", "instruments": ["vocals"]},
            }
        )

        model = get_model_from_config("bs_roformer", config)
        first_mask_linear = model.mask_estimators[0].to_freqs[0][0][0]

        assert tuple(first_mask_linear.weight.shape) == (16, 8)

    def test_hyperace_variation_selects_segment_head(self):
        """Registry metadata can select the HyperACE MaskEstimator variation."""
        config = ConfigDict(
            {
                "model": {
                    "dim": 8,
                    "depth": 1,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "freqs_per_bands": [512, 513],
                    "dim_head": 4,
                    "heads": 1,
                    "dim_freqs_in": 1025,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 512,
                    "stft_win_length": 2048,
                    "mask_estimator_depth": 2,
                    "mlp_expansion_factor": 4,
                },
                "training": {"target_instrument": "vocals", "instruments": ["vocals"]},
            }
        )

        model = get_model_from_config(
            "bs_roformer",
            config,
            model_variation="hyperace",
        )

        assert model.mask_estimator_variant == "hyperace"
        assert hasattr(model.mask_estimators[0], "segm")

    def test_fno_variation_selects_fourier_head(self):
        """Registry metadata can select the FNO MaskEstimator variation."""
        config = ConfigDict(
            {
                "model": {
                    "dim": 8,
                    "depth": 1,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "freqs_per_bands": [512, 513],
                    "dim_head": 4,
                    "heads": 1,
                    "dim_freqs_in": 1025,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 512,
                    "stft_win_length": 2048,
                    "mask_estimator_depth": 2,
                    "mlp_expansion_factor": 4,
                },
                "training": {"target_instrument": "other", "instruments": ["vocals", "other"]},
            }
        )

        model = get_model_from_config(
            "bs_roformer",
            config,
            model_variation="fno",
        )

        assert model.mask_estimator_variant == "fno"
        assert hasattr(model.mask_estimators[0].to_freqs[0][0], "fno_blocks")

    def test_checkpoint_loader_strips_extra_metadata(self, tmp_path, monkeypatch):
        """FNO checkpoint metadata is loader compatibility data, not a model key."""
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_bytes(b"placeholder")

        monkeypatch.setattr(
            "bs_roformer.utils.torch.load",
            lambda *_args, **_kwargs: {"weight": object(), "_metadata": object()},
        )

        state = load_checkpoint_state(checkpoint)

        assert set(state) == {"weight"}


def main():
    """Run tests directly without pytest."""
    print("=" * 60)
    print("Testing all model configs")
    print("=" * 60)

    configs = find_all_configs()
    if not configs:
        print("No configs found!")
        return 1

    print(f"\nFound {len(configs)} config(s):\n")

    passed = 0
    failed = 0
    skipped = 0

    for name, config_path in configs:
        print(f"Testing: {name}")

        # Test 1: Load with safe_load
        try:
            config = load_config_safe(config_path)
            print("  [OK] Config loads with safe_load")
        except Exception as e:
            print(f"  [FAIL] Failed to load: {e}")
            failed += 1
            continue

        # Test 2: Check param type
        param = config.model.get("multi_stft_resolutions_window_sizes")
        if param is not None:
            if isinstance(param, list):
                print("  [OK] multi_stft_resolutions_window_sizes is list (will be converted)")
            else:
                print(f"  [WARN] Unexpected type: {type(param).__name__}")

        # Test 3: Model instantiation
        if not hasattr(config, "training"):
            print("  [SKIP] Skipped model test (missing 'training' section)")
            skipped += 1
            continue

        try:
            model = get_model_from_config("bs_roformer", config)
            if model is not None:
                print("  [OK] Model instantiated successfully")
                passed += 1
            else:
                print("  [FAIL] Model is None")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Model instantiation failed: {e}")
            failed += 1

        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
