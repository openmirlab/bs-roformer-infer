"""Default SW checkpoint parity with pristine historical BS-RoFormer output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

from bs_roformer.checkpoints import checkpoint_metadata
from bs_roformer.clean_api import BSRoformerSession
from bs_roformer.inference import SafeLoaderWithTuple
from bs_roformer.utils import get_model_from_config, load_checkpoint_state


FIXTURES = Path(__file__).parent / "fixtures/original_default_golden"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
MODEL_NAME = "roformer-model-bs-roformer-sw-by-jarredou"
MODEL_DIR = Path("~/.cache/bs-roformer-infer").expanduser() / MODEL_NAME


def test_original_reference_and_registry_metadata():
    artifacts = checkpoint_metadata(MODEL_NAME)["artifacts"]
    for kind, expected in (
        ("checkpoint", MANIFEST["checkpoint_sha256"]),
        ("config", MANIFEST["config_sha256"]),
    ):
        assert next(a["sha256"] for a in artifacts if a["kind"] == kind) == expected
    with np.load(FIXTURES / "two_seconds.npz") as golden:
        assert list(golden["waveform"].shape) == MANIFEST["waveform_shape"]
        for mode in ("output_fp32", "output_amp"):
            assert list(golden[mode].shape) == MANIFEST["output_shape"]
            assert np.isfinite(golden[mode]).all()


def _real_assets():
    if not torch.cuda.is_available():
        pytest.skip("real default SW checkpoint requires CUDA")
    if (
        torch.__version__ != MANIFEST["torch"]
        or torch.cuda.get_device_name(0) != MANIFEST["cuda_device"]
        or list(torch.cuda.get_device_capability(0)) != MANIFEST["cuda_capability"]
    ):
        pytest.skip("original output requires the recorded Torch/CUDA profile")
    checkpoint = MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt"
    config_file = MODEL_DIR / "BS-Rofo-SW-Fixed.yaml"
    if not checkpoint.is_file() or not config_file.is_file():
        pytest.skip("official default SW checkpoint and config are not cached")
    return checkpoint, config_file


@pytest.mark.realweights
def test_port_matches_pristine_original_complete_outputs():
    checkpoint, config_file = _real_assets()

    torch.set_num_threads(1)
    torch.manual_seed(MANIFEST["seed"])
    config = ConfigDict(yaml.load(config_file.read_text(), Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config).eval().to("cuda:0")
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    with np.load(FIXTURES / "two_seconds.npz") as golden:
        audio = torch.from_numpy(golden["waveform"]).to("cuda:0")
        with torch.no_grad():
            fp32 = model(audio).cpu().numpy()
            with torch.autocast("cuda"):
                amp = model(audio).cpu().numpy()
        assert list(fp32.shape) == MANIFEST["output_shape"]
        np.testing.assert_array_equal(fp32, golden["output_fp32"])
        np.testing.assert_array_equal(amp, golden["output_amp"])


@pytest.mark.realweights
def test_public_session_writes_complete_baseline_stems(tmp_path):
    checkpoint, config_file = _real_assets()
    torch.set_num_threads(1)
    torch.manual_seed(MANIFEST["seed"])
    with np.load(FIXTURES / "thirteen_seconds_public.npz") as golden:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        sf.write(inputs / "synthetic.wav", golden["waveform"],
                 MANIFEST["public_baseline"]["sample_rate"], subtype="FLOAT")
        with BSRoformerSession(
            model_path=checkpoint, config_path=config_file, device="cuda:0", progress=False
        ) as session:
            written = session.infer(inputs, store_dir=tmp_path / "outputs")
        assert {entry.output_id for entry in written.outputs} == set(
            MANIFEST["public_baseline"]["output_ids"]
        )
        for entry in written.outputs:
            actual, sample_rate = sf.read(entry.output_path, dtype="float32",
                                          always_2d=True)
            assert sample_rate == MANIFEST["public_baseline"]["sample_rate"]
            assert list(actual.shape) == MANIFEST["public_baseline"]["output_shape"]
            np.testing.assert_array_equal(actual, golden[entry.output_id])
