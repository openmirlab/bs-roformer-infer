"""Inference output-manifest regressions for the clean API and default model metadata.

These tests stay offline by monkeypatching soundfile and demix_track, so they prove
the manifest contract without touching model downloads or real checkpoints. They also
lock the pinned SW model's six raw stems separately from the derived instrumental,
which belongs to inference output metadata rather than BSModel.default_sources.

Reads: bs_roformer.inference (OutputManifest, run_folder), bs_roformer.clean_api
(BSRoformerSession), bs_roformer.model_registry (DEFAULT_MODEL, MODEL_REGISTRY)
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

import bs_roformer.inference as inference_module
from bs_roformer.clean_api import BSRoformerSession
from bs_roformer.inference import OutputFile, OutputManifest
from bs_roformer.model_registry import DEFAULT_MODEL, MODEL_REGISTRY


class _DummyModel:
    def eval(self):
        return self


def _multi_stem_config() -> ConfigDict:
    return ConfigDict(
        {
            "training": {
                "instruments": ["bass", "drums", "other", "vocals", "guitar", "piano"],
                "target_instrument": None,
            },
            "inference": {
                "chunk_size": 8,
                "num_overlap": 2,
            },
        }
    )


def test_default_model_default_sources_lock_six_raw_stems():
    model = MODEL_REGISTRY.get(DEFAULT_MODEL)
    assert model.category == "multi-stem"
    assert model.default_sources == ["bass", "drums", "other", "vocals", "guitar", "piano"]
    assert "instrumental" not in model.default_sources


def test_run_folder_returns_manifest_for_every_written_output(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    alpha = input_dir / "alpha.wav"
    beta = input_dir / "beta.wav"
    alpha.write_bytes(b"")
    beta.write_bytes(b"")

    alpha_mix = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    beta_mix = np.array([[7.0, 9.0], [11.0, 13.0]], dtype=np.float32)
    audio_by_path = {
        alpha: alpha_mix,
        beta: beta_mix,
    }
    writes: list[tuple[Path, np.ndarray]] = []

    def fake_read(path):
        return audio_by_path[Path(path)], 44100

    def fake_write(path, data, sr, subtype):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stub")
        writes.append((target, np.asarray(data).copy()))
        assert sr == 44100
        assert subtype == "FLOAT"

    def fake_demix_track(config, model, mixture, device, first_chunk_time):
        sources = {}
        for index, output_id in enumerate(config.training.instruments, start=1):
            sources[output_id] = np.full(mixture.shape, float(index), dtype=np.float32)
        return sources, 0.01 if first_chunk_time is None else first_chunk_time

    monkeypatch.setattr(inference_module.sf, "read", fake_read)
    monkeypatch.setattr(inference_module.sf, "write", fake_write)
    monkeypatch.setattr(inference_module, "demix_track", fake_demix_track)
    monkeypatch.setattr(inference_module.time, "sleep", lambda seconds: None)

    manifest = inference_module.run_folder(
        _DummyModel(),
        Namespace(input_folder=input_dir, store_dir=tmp_path / "outputs"),
        _multi_stem_config(),
        device="cpu",
        verbose=True,
    )

    assert isinstance(manifest, OutputManifest)
    assert len(manifest.outputs) == 14
    assert all(Path(output.output_path).exists() for output in manifest.outputs)
    assert manifest.as_dict() == {
        "outputs": [
            {
                "input_path": output.input_path,
                "output_id": output.output_id,
                "output_path": output.output_path,
            }
            for output in manifest.outputs
        ]
    }

    alpha_outputs = [output for output in manifest.outputs if output.input_path == str(alpha)]
    beta_outputs = [output for output in manifest.outputs if output.input_path == str(beta)]
    assert [output.output_id for output in alpha_outputs] == [
        "bass",
        "drums",
        "other",
        "vocals",
        "guitar",
        "piano",
        "instrumental",
    ]
    assert [output.output_id for output in beta_outputs] == [
        "bass",
        "drums",
        "other",
        "vocals",
        "guitar",
        "piano",
        "instrumental",
    ]

    write_map = {path.name: data for path, data in writes}
    np.testing.assert_allclose(
        write_map["alpha_instrumental.wav"],
        alpha_mix - 4.0,
    )
    np.testing.assert_allclose(
        write_map["beta_instrumental.wav"],
        beta_mix - 4.0,
    )


def test_session_infer_returns_run_folder_manifest(monkeypatch, tmp_path):
    expected = OutputManifest(
        outputs=(
            OutputFile(
                input_path=str(tmp_path / "inputs" / "song.wav"),
                output_id="vocals",
                output_path=str(tmp_path / "outputs" / "song_vocals.wav"),
            ),
        )
    )

    def fake_run_folder(model, args, config, device, verbose=False):
        assert args.input_folder == tmp_path / "inputs"
        assert args.store_dir == tmp_path / "outputs"
        assert verbose is True
        return expected

    monkeypatch.setattr(inference_module, "run_folder", fake_run_folder)

    session = BSRoformerSession(model=_DummyModel(), config=_multi_stem_config(), device="cpu")
    result = session.infer(tmp_path / "inputs", store_dir=tmp_path / "outputs", verbose=True)

    assert result is expected
