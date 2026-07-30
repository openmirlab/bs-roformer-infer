"""Offline lifecycle and device contract tests."""
from argparse import Namespace
import pytest
import torch
from bs_roformer.inference import _select_device
from bs_roformer.clean_api import BSRoformerSession


def test_device_contract(monkeypatch):
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("bs_roformer.inference.mps_available", lambda: False)
    assert _select_device(Namespace(device=None)) == torch.device("cpu")
    assert _select_device(Namespace(device="auto")) == torch.device("cpu")
    with pytest.raises(RuntimeError):
        _select_device(Namespace(device="cuda:1"))
    with pytest.raises(ValueError):
        _select_device(Namespace(device="tpu"))
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.device_count", lambda: 2)
    assert _select_device(Namespace(device="cuda:1")) == torch.device("cuda:1")


def test_explicit_mps_resolves_when_available(monkeypatch):
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("bs_roformer.inference.mps_available", lambda: True)
    assert _select_device(Namespace(device="mps")) == torch.device("mps")


def test_explicit_mps_raises_when_unavailable(monkeypatch):
    """An explicit accelerator is honoured or the call fails -- never downgraded."""
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("bs_roformer.inference.mps_available", lambda: False)
    with pytest.raises(RuntimeError):
        _select_device(Namespace(device="mps"))


def test_auto_never_promotes_to_mps(monkeypatch):
    """Legacy auto-selection is preserved: MPS is opt-in, so existing Mac callers
    keep the exact compute path -- and the exact outputs -- they had before."""
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("bs_roformer.inference.mps_available", lambda: True)
    assert _select_device(Namespace(device=None)) == torch.device("cpu")
    assert _select_device(Namespace(device="auto")) == torch.device("cpu")


def test_session_lifecycle_cache_resolver_and_cuda_forwarding(monkeypatch, tmp_path):
    checkpoint, config = tmp_path / "model.ckpt", tmp_path / "model.yaml"
    checkpoint.write_bytes(b"x")
    config.write_text("model: {}\n")
    resolver_calls, constructed, seen = [], [], {}

    def resolver(*_args, **kwargs):
        resolver_calls.append(kwargs)
        return checkpoint, config

    class FakeModel:
        def load_state_dict(self, _state):
            pass

        def to(self, device):
            seen["to"] = device
            return self

        def eval(self):
            return self

        def cpu(self):
            return self

    monkeypatch.setattr("bs_roformer.download.ensure_model_assets", resolver)
    monkeypatch.setattr("bs_roformer.clean_api.yaml.load", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "bs_roformer.utils.get_model_from_config",
        lambda *_a, **_k: constructed.append(1) or FakeModel(),
    )
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {})
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("bs_roformer.inference.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr(
        "bs_roformer.inference.separate_folder_with",
        lambda separate, _a, _c, **_k: seen.setdefault("run", separate),
    )
    session = BSRoformerSession(model_name="unknown", device="cuda:1")
    with pytest.raises(RuntimeError):
        session.infer(tmp_path)
    session.cache_info()
    assert len(resolver_calls) == 1
    session.load().load()
    assert len(resolver_calls) == 2 and len(constructed) == 1
    assert seen["to"] == torch.device("cuda:1")
    session.infer(tmp_path)
    assert callable(seen["run"])
    # The explicitly requested device still reaches the compute path and is
    # reported back on a public surface rather than only on the model.
    assert session.cache_info()["device"] == "cuda:1"
    assert session.cache_info()["backend"] == "torch"
    session.release()
    session.load()
    assert len(constructed) == 2
    session.close()
    session.close()
    with pytest.raises(RuntimeError):
        session.load()
