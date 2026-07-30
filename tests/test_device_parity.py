"""Real-checkpoint CPU-vs-MPS output parity (article 2's accuracy gate).

Marked ``realweights`` and deselected by default: it needs the real default
checkpoint already present on disk and an Apple Silicon Mac. It never downloads
-- a test that silently fetches ~700MB is not an offline test.

Run explicitly:  pytest -m realweights tests/test_device_parity.py -v

Recorded measurement (2026-07-30, Apple M2, torch 2.13.0, 5 s seeded stereo,
BS-Rofo-SW-Fixed.ckpt) -- worst across all six stems:

    max_abs = 1.136e-07   rel_to_peak = 4.5e-06   (stem "other")

The assertion threshold sits well above that so ordinary float noise cannot flake
it, while a genuinely broken MPS path -- which diverges by orders of magnitude,
not by a factor of two -- still fails loudly.
"""
import numpy as np
import pytest
import torch
import yaml
from ml_collections import ConfigDict

from bs_roformer.download import ensure_model_assets
from bs_roformer.inference import SafeLoaderWithTuple, _select_device, mps_available
from bs_roformer.utils import demix_track, get_model_from_config, load_checkpoint_state

pytestmark = pytest.mark.realweights

MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
SEED = 20260730
SECONDS = 5
MAX_ABS_TOLERANCE = 1e-5


def _cached_assets():
    try:
        return ensure_model_assets(MODEL, download_missing=False)
    except (FileNotFoundError, KeyError):
        return None


def test_cpu_and_mps_agree_on_real_checkpoint():
    if not mps_available():
        pytest.skip("MPS unavailable: needs an Apple Silicon Mac and an arm64 torch build")
    assets = _cached_assets()
    if assets is None:
        pytest.skip(f"{MODEL} is not in the local cache; run bs-roformer-download first")
    checkpoint, config_path = assets

    with open(config_path) as handle:
        config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(load_checkpoint_state(checkpoint, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(SEED)
    mix = torch.from_numpy(
        (rng.standard_normal((2, SECONDS * 44100)) * 0.1).astype(np.float32)
    )

    results = {}
    for name in ("cpu", "mps"):
        device = _select_device(_DeviceRequest(name))
        results[name], _ = demix_track(config, model.to(device), mix, device)

    assert set(results["cpu"]) == set(results["mps"])
    for stem, reference in results["cpu"].items():
        candidate = results["mps"][stem]
        assert reference.shape == candidate.shape, f"{stem}: shape moved between devices"
        worst = float(np.abs(reference - candidate).max())
        assert worst < MAX_ABS_TOLERANCE, (
            f"{stem}: CPU-vs-MPS max abs {worst:.3e} exceeds {MAX_ABS_TOLERANCE:.0e} "
            f"(torch {torch.__version__}); investigate rather than widen the tolerance"
        )


class _DeviceRequest:
    """Minimal stand-in for the argparse Namespace _select_device reads."""

    def __init__(self, device):
        self.device = device
