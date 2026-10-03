"""Capture default SW model output from pristine historical BS-RoFormer.

Set BS_ROFORMER_ORIGINAL_DIR to the read-only upstream checkout at the pinned
revision. This script imports upstream model code only, not the port package.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml


UPSTREAM_REVISION = "93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf"
CHECKPOINT_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
CONFIG_SHA256 = "f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b"
MODEL_DIR = Path("~/.cache/bs-roformer-infer/roformer-model-bs-roformer-sw-by-jarredou").expanduser()
ORIGINAL_PARAMETERS = {
    "dim", "depth", "stereo", "num_stems", "time_transformer_depth",
    "freq_transformer_depth", "freqs_per_bands", "dim_head", "heads",
    "attn_dropout", "ff_dropout", "flash_attn", "dim_freqs_in",
    "stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized",
    "mask_estimator_depth", "multi_stft_resolution_loss_weight",
    "multi_stft_resolutions_window_sizes", "multi_stft_hop_size",
    "multi_stft_normalized",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Default SW golden requires CUDA")
    upstream = Path(os.environ["BS_ROFORMER_ORIGINAL_DIR"]).resolve()
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {head}")
    changes = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changes:
        raise ValueError("Original source has tracked changes")
    checkpoint = MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt"
    config_file = MODEL_DIR / "BS-Rofo-SW-Fixed.yaml"
    if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(config_file) != CONFIG_SHA256:
        raise ValueError("Default SW official asset SHA-256 mismatch")
    sys.path.insert(0, str(upstream))
    from bs_roformer import BSRoformer

    config = yaml.load(config_file.read_text(), Loader=yaml.FullLoader)
    kwargs = {key: value for key, value in config["model"].items()
              if key in ORIGINAL_PARAMETERS}
    torch.set_num_threads(1)
    torch.manual_seed(12345)
    model = BSRoformer(**kwargs).eval().to("cuda:0")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("_metadata", None)
    model.load_state_dict(state, strict=True)

    rate = 44100
    time = np.arange(rate * 2, dtype=np.float64) / rate
    left = 0.2 * np.sin(2 * np.pi * 220 * time) * np.exp(-0.7 * time)
    right = 0.15 * np.sin(2 * np.pi * 330 * time) * np.exp(-0.5 * time)
    waveform = np.stack([left, right]).astype(np.float32)[None]
    audio = torch.from_numpy(waveform).to("cuda:0")
    with torch.no_grad():
        output_fp32 = model(audio).cpu().numpy()
        with torch.autocast("cuda"):
            output_amp = model(audio).cpu().numpy()
    assert np.isfinite(output_fp32).all() and np.isfinite(output_amp).all()

    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/original_default_golden"
    fixtures.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixtures / "two_seconds.npz", waveform=waveform,
                        output_fp32=output_fp32, output_amp=output_amp)
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": sha256(upstream / "bs_roformer/bs_roformer.py"),
        "upstream_attention_sha256": sha256(upstream / "bs_roformer/attend.py"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "seed": 12345,
        "sample_rate": rate,
        "waveform_shape": list(waveform.shape),
        "output_shape": list(output_fp32.shape),
    }
    (fixtures / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Original model: {waveform.shape} -> {output_fp32.shape}")


if __name__ == "__main__":
    main()
