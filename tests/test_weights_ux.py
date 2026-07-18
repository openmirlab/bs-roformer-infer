"""Weights-UX regression tests -- sha256 verification wiring + models-dir resolution.

Locks the 0.1.4 weights contract: downloads verify against data/checksums.json
(a wrong hash must fail and remove the file, not ship it), the models directory
resolves explicit arg > $BS_ROFORMER_MODELS_PATH > ~/.cache/bs-roformer-infer with
legacy ./models as a read fallback, and ensure_model_assets() prefers existing
local copies without touching the network. All tests run offline on temp files;
the fake-response test drives download_file end-to-end with monkeypatched requests.
is_model_cached() gets its own section: it must agree with ensure_model_assets()
on what counts as cached (default dir, legacy dir, explicit override) without ever
downloading -- it exists precisely so a caller can ask "is it cached?" for status
reporting without risking a side-effecting download.

Reads: bs_roformer.download (get_file_hash, verify_file_integrity, download_file,
CHECKSUMS, default_models_dir, models_search_dirs, ensure_model_assets,
is_model_cached), bs_roformer.model_registry (MODEL_REGISTRY, DEFAULT_MODEL)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from bs_roformer import download as dl
from bs_roformer.model_registry import DEFAULT_MODEL, MODEL_REGISTRY

HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------- hashing


def test_get_file_hash_matches_hashlib(tmp_path):
    payload = b"bs-roformer weights ux test payload"
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    assert dl.get_file_hash(f) == hashlib.sha256(payload).hexdigest()


def test_verify_integrity_accepts_matching_sha256(tmp_path):
    payload = b"good bytes"
    f = tmp_path / "ok.bin"
    f.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert dl.verify_file_integrity(f, expected_size=len(payload), expected_sha256=digest)


def test_verify_integrity_rejects_sha256_mismatch(tmp_path):
    f = tmp_path / "bad.bin"
    f.write_bytes(b"corrupted bytes")
    assert not dl.verify_file_integrity(f, expected_sha256="0" * 64)


def test_verify_integrity_rejects_size_mismatch(tmp_path):
    f = tmp_path / "short.bin"
    f.write_bytes(b"123")
    assert not dl.verify_file_integrity(f, expected_size=999)


def test_verify_integrity_rejects_empty_and_missing(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert not dl.verify_file_integrity(empty)
    assert not dl.verify_file_integrity(tmp_path / "missing.bin")


# ---------------------------------------------------------------- checksum registry


def test_checksums_file_well_formed():
    assert dl.CHECKSUMS, "data/checksums.json must ship with recorded hashes"
    for filename, entry in dl.CHECKSUMS.items():
        assert HEX64.fullmatch(entry["sha256"]), f"bad sha256 for {filename}"
        assert isinstance(entry["size"], int) and entry["size"] > 0, f"bad size for {filename}"


def test_default_model_checkpoint_has_recorded_checksum():
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    info = dl.CHECKSUMS.get(entry.checkpoint)
    assert info is not None, "the default model's checkpoint must have a recorded sha256"
    # Locked value: verified 2026-07-12 against the enerjazzer HF LFS pointer and
    # two independently downloaded local copies.
    assert info["sha256"] == "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
    assert info["size"] == 699412152


# ---------------------------------------------------------------- models-dir resolution


def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path / "weights"))
    assert dl.default_models_dir() == tmp_path / "weights"


def test_models_dir_cache_default(monkeypatch):
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    assert dl.default_models_dir() == Path.home() / ".cache" / dl.CACHE_DIR_NAME


def test_explicit_models_dir_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path / "env-dir"))
    dirs = dl.models_search_dirs(tmp_path / "explicit")
    assert dirs[0] == tmp_path / "explicit"


def test_legacy_models_dir_is_search_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    dirs = dl.models_search_dirs()
    assert dirs[0] == Path.home() / ".cache" / dl.CACHE_DIR_NAME
    assert Path("models") in dirs


def test_no_legacy_fallback_without_models_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert dl.models_search_dirs() == [Path.home() / ".cache" / dl.CACHE_DIR_NAME]


# ---------------------------------------------------------------- ensure_model_assets


def test_ensure_model_assets_prefers_existing_local_copy(monkeypatch, tmp_path):
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    model_dir = tmp_path / entry.slug
    model_dir.mkdir(parents=True)
    (model_dir / entry.checkpoint).write_bytes(b"fake checkpoint")

    def _no_network(*args, **kwargs):
        pytest.fail("network download attempted despite local copy")

    monkeypatch.setattr(dl, "download_file", _no_network)
    ckpt, cfg = dl.ensure_model_assets(DEFAULT_MODEL, models_dir=tmp_path)
    assert ckpt == model_dir / entry.checkpoint
    assert cfg == model_dir / entry.config
    assert cfg.exists(), "packaged config should be copied in without network"


def test_ensure_model_assets_finds_legacy_models_dir(monkeypatch, tmp_path):
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    legacy_dir = tmp_path / "models" / entry.slug
    legacy_dir.mkdir(parents=True)
    (legacy_dir / entry.checkpoint).write_bytes(b"fake checkpoint")
    (legacy_dir / entry.config).write_text("training: {}\n")

    monkeypatch.setattr(dl, "download_file",
                        lambda *a, **k: pytest.fail("network download attempted"))
    ckpt, cfg = dl.ensure_model_assets(DEFAULT_MODEL)
    assert ckpt == Path("models") / entry.slug / entry.checkpoint
    assert cfg == Path("models") / entry.slug / entry.config


def test_ensure_model_assets_raises_when_missing_and_download_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError):
        dl.ensure_model_assets(DEFAULT_MODEL, models_dir=tmp_path, download_missing=False)


# ---------------------------------------------------------------- is_model_cached


def test_is_model_cached_true_in_default_dir(monkeypatch, tmp_path):
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path))
    model_dir = tmp_path / entry.slug
    model_dir.mkdir(parents=True)
    (model_dir / entry.checkpoint).write_bytes(b"fake checkpoint")

    assert dl.is_model_cached(DEFAULT_MODEL) is True


def test_is_model_cached_true_in_legacy_dir(monkeypatch, tmp_path):
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    monkeypatch.delenv(dl.MODELS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    legacy_dir = tmp_path / "models" / entry.slug
    legacy_dir.mkdir(parents=True)
    (legacy_dir / entry.checkpoint).write_bytes(b"fake checkpoint")

    assert dl.is_model_cached(DEFAULT_MODEL) is True


def test_is_model_cached_false_when_absent_everywhere(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)

    assert dl.is_model_cached(DEFAULT_MODEL) is False


def test_is_model_cached_respects_explicit_models_dir_override(monkeypatch, tmp_path):
    entry = MODEL_REGISTRY.get(DEFAULT_MODEL)
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path / "env-dir"))
    explicit_dir = tmp_path / "explicit"
    model_dir = explicit_dir / entry.slug
    model_dir.mkdir(parents=True)
    (model_dir / entry.checkpoint).write_bytes(b"fake checkpoint")

    # Env dir has nothing -- only the explicit override does.
    assert dl.is_model_cached(DEFAULT_MODEL, models_dir=explicit_dir) is True
    assert dl.is_model_cached(DEFAULT_MODEL) is False


def test_is_model_cached_never_touches_network(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dl, "download_file",
                        lambda *a, **k: pytest.fail("network download attempted"))

    assert dl.is_model_cached(DEFAULT_MODEL) is False


# ---------------------------------------------------------------- download verification wiring


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield self._payload


def test_download_file_rejects_wrong_sha256(monkeypatch, tmp_path):
    payload = b"streamed model bytes"
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(payload))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    target = tmp_path / "model.ckpt"
    ok = dl.download_file("http://example.invalid/model.ckpt", target,
                          expected_sha256="0" * 64, max_retries=2)
    assert not ok
    assert not target.exists(), "corrupt download must be removed, not kept"


def test_download_file_accepts_matching_sha256(monkeypatch, tmp_path):
    payload = b"streamed model bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(payload))
    target = tmp_path / "model.ckpt"
    ok = dl.download_file("http://example.invalid/model.ckpt", target,
                          expected_size=len(payload), expected_sha256=digest)
    assert ok
    assert target.read_bytes() == payload
