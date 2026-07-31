"""Backend seam contract: resolution, import purity, and honest failure.

These are offline and hardware-independent. They guard the two properties the
seam exists to protect -- that a requested backend is honoured or refused, never
silently swapped, and that the default import path stays free of optional
frameworks.
"""
import argparse
import importlib.util
import subprocess
import sys

import pytest

from bs_roformer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    get_backend,
    resolve_backend_name,
)


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_falls_back_to_torch_when_nothing_accelerated_is_installed():
    """`auto` is the one place a fallback is what the caller asked for."""
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("cuda")


def test_unavailable_backend_raises_rather_than_substituting():
    """An explicit request is honoured or fails loudly -- never downgraded.

    A silent substitution is only ever discovered by noticing the wrong hardware
    was busy, which is exactly the failure article 4b forbids for devices.

    Only meaningful where MLX is genuinely absent: with the [mlx] extra
    installed the request succeeds, which is the correct behaviour, not a
    regression. Guarded so a shared MLX-capable env cannot report this as a
    failure.
    """
    if _mlx_installed():
        pytest.skip("MLX is installed here; this asserts the unavailable path")
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release", "resolved_device"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_refuses_a_variation_it_has_no_head_for():
    """Refusing beats mis-running.

    Upstream's MLX model swallows unknown constructor arguments, so an unsupported
    mask-estimator variation would silently build a plain MaskEstimator and emit
    confident garbage rather than fail.
    """
    from bs_roformer.backends.mlx_backend import MLXBackend

    with pytest.raises(BackendUnavailable) as excinfo:
        MLXBackend.assert_supports_variation("no_such_head")
    assert "no_such_head" in str(excinfo.value)


@pytest.mark.parametrize("variation", [None, "", "mlp"])
def test_mlx_accepts_the_stock_mask_estimator(variation):
    from bs_roformer.backends.mlx_backend import MLXBackend

    MLXBackend.assert_supports_variation(variation)


def test_mlx_variation_support_is_measured_not_declared():
    """Capability comes from which head modules exist, so a head that has not been
    written cannot be advertised and then fail deep inside construction."""
    from bs_roformer.backends.mlx_backend import supported_variations
    from bs_roformer.mlx.heads import VARIANTS, available_variants

    available = available_variants()
    assert "mlp" in available
    assert set(available) <= set(VARIANTS)
    assert set(available) <= supported_variations()


def test_every_registry_variation_has_an_mlx_head():
    """The 24/24 goal, asserted rather than assumed: every mask-estimator variation
    any registry checkpoint declares must be buildable under MLX."""
    from pathlib import Path

    import tomllib

    from bs_roformer.mlx.heads import available_variants

    registry = Path("src/bs_roformer/config/checkpoints.toml")
    if not registry.exists():
        pytest.skip("registry TOML not found from this working directory")
    models = tomllib.loads(registry.read_text())["models"]
    declared = {m.get("variation") for m in models.values()} - {None, ""}
    missing = sorted(declared - set(available_variants()))
    assert not missing, f"registry variations with no MLX head: {missing}"


def test_mlx_refuses_chunking_that_is_not_hop_aligned():
    """The chunked path's length bookkeeping assumes chunk_size % hop == 0."""
    from ml_collections import ConfigDict

    from bs_roformer.backends.mlx_backend import MLXBackend

    config = ConfigDict(
        {
            "inference": {"chunk_size": 588801, "num_overlap": 2},
            "model": {"stft_hop_length": 512},
        }
    )
    with pytest.raises(BackendUnavailable):
        MLXBackend._reject_unaligned_chunking(config)


def test_auto_skips_a_backend_that_cannot_run_this_checkpoint(monkeypatch):
    """`auto` means "pick one that works", so it must not settle on a backend that
    has no head for this checkpoint and then fail at construction -- Torch would
    simply have worked."""
    from bs_roformer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    assert resolve_backend_name("auto", variation=None) == "mlx"
    assert resolve_backend_name("auto", variation="no_such_head") == "torch"


def test_explicit_backend_is_never_downgraded_for_an_unsupported_checkpoint(monkeypatch):
    """An explicit request is honoured or refused -- `auto`'s fallback is not it."""
    from bs_roformer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx", variation="no_such_head")


@pytest.mark.parametrize("device", [None, "auto", "mps"])
def test_mlx_accepts_only_its_own_execution_target(device):
    from bs_roformer.backends.mlx_backend import MLXBackend

    assert MLXBackend._select_device(device) == "mps"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_mlx_refuses_a_torch_device_rather_than_reinterpreting_it(device):
    """architecture.md failure mode 2, now actually enforced.

    Treating device="cuda" as "the Apple GPU anyway" would discard what the
    caller explicitly asked for.
    """
    from bs_roformer.backends.mlx_backend import MLXBackend

    with pytest.raises(BackendUnavailable):
        MLXBackend._select_device(device)


def test_cli_builds_a_non_torch_backend_from_the_checkpoint(monkeypatch, tmp_path):
    """The CLI path must not hand a constructed torch module to another framework.

    It used to do exactly that: build the Torch model unconditionally, then pass
    it to whichever backend was resolved, so `--backend mlx` produced a backend
    holding the wrong framework's model. No test covered it.
    """
    from bs_roformer import inference as inference_module

    seen = {}

    class FakeBackend:
        name = "fake"

        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def supports_variation(cls, variation):
            return True

        @classmethod
        def from_checkpoint(cls, **kwargs):
            seen.update(kwargs)
            return cls()

        def separate(self, mix):  # pragma: no cover - never reached
            raise AssertionError("not exercised")

    monkeypatch.setattr("bs_roformer.backends._load", lambda name: FakeBackend)
    monkeypatch.setattr(
        inference_module, "get_model_from_config", _fail_if_called("torch model built")
    )
    monkeypatch.setattr(
        inference_module, "separate_folder_with", lambda *a, **k: "manifest"
    )
    monkeypatch.setattr(
        inference_module, "_resolve_model_assets", lambda args, parser: None
    )
    monkeypatch.setattr(inference_module.yaml, "load", lambda *a, **k: {})

    config_path = tmp_path / "c.yaml"
    config_path.write_text("model: {}\n")
    args = _CliArgs(config_path=config_path, model_path=tmp_path / "m.ckpt")

    assert inference_module.proc_folder(args) == "manifest"
    assert seen["checkpoint_path"] == tmp_path / "m.ckpt"


def _fail_if_called(what):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{what}: the non-torch path must not do this")

    return _raise


class _CliArgs(argparse.Namespace):
    """Minimal stand-in for the parsed CLI namespace proc_folder consumes.

    Must actually be a Namespace: proc_folder isinstance-checks it to decide
    whether to parse or use as-is.
    """

    def __init__(self, **kwargs):
        self.backend = "mlx"
        self.device = None
        self.device_ids = None
        self.model_type = "bs_roformer"
        self.model_variation = None
        self.input_folder = "in"
        self.store_dir = "out"
        self.__dict__.update(kwargs)


def _mlx_installed() -> bool:
    return importlib.util.find_spec("mlx") is not None


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install bs-roformer-infer` must stay MLX-free and import-clean.

    Runs in a SUBPROCESS on purpose. In-process this test is order-dependent:
    any earlier test that legitimately calls MLXBackend.is_available() imports
    mlx as a side effect, so a clean import path reads as a leak (and, worse, a
    genuine leak could be masked by import caching in the other direction). It
    passed alone and failed in-suite before this fix -- a test that can be wrong
    in both directions guards nothing.
    """
    probe = (
        "import sys, bs_roformer; "
        "optional = {'mlx', 'mlx_spectro', 'mlx_audio_io', 'mlx_audio_separator'}; "
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import bs_roformer pulled in optional frameworks: {leaked}"
