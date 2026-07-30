"""Torch-vs-MLX output parity on the real checkpoint, including silence.

Marked ``realweights`` and deselected by default: needs the MLX extra, an Apple
Silicon Mac, and the default checkpoint already on disk. It never downloads.

Run explicitly:  pytest -m realweights tests/test_mlx_parity.py -v

The silence case is the point of this file. A chunk of clean signal agreed to
4.0e-07 while a chunk with a zero-padded tail diverged by 1.455e-02 -- and every
track's final chunk is padded, so shipping on the clean-signal number alone would
have shipped a wrong answer that benchmarks beautifully. Root cause and the
workaround it guards live in ``bs_roformer/mlx/model.py::exact_zero_safe_rfft``.

Recorded measurement (2026-07-30, Apple M2, torch 2.13.0, mlx 0.31.2):

    no silence          2.831e-07
    zero-padded tail    2.226e-07   (1.455e-02 without the workaround)
    near-silent tail    1.937e-07
"""
import numpy as np
import pytest
import torch
import yaml
from ml_collections import ConfigDict

from bs_roformer.backends.base import ChunkingPlan
from bs_roformer.download import ensure_model_assets
from bs_roformer.inference import SafeLoaderWithTuple, _select_device, mps_available
from bs_roformer.utils import get_model_from_config, load_checkpoint_state

pytestmark = pytest.mark.realweights

MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
SEED = 1
SIGNAL_SAMPLES = 220500
MAX_ABS_TOLERANCE = 1e-5


def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def models():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'bs-roformer-infer[mlx]'")
    if not mps_available():
        pytest.skip("needs an Apple Silicon Mac and an arm64 torch build")
    try:
        checkpoint, config_path = ensure_model_assets(MODEL, download_missing=False)
    except (FileNotFoundError, KeyError):
        pytest.skip(f"{MODEL} is not cached; run bs-roformer-download first")

    from bs_roformer.backends.mlx_backend import MLXBackend

    with open(config_path) as handle:
        config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))

    torch_model = get_model_from_config("bs_roformer", config)
    torch_model.load_state_dict(load_checkpoint_state(checkpoint, map_location="cpu"))
    device = _select_device(_DeviceRequest("mps"))
    torch_model = torch_model.to(device).eval()

    backend = MLXBackend.from_checkpoint(
        config=config, checkpoint_path=checkpoint, variation=None
    )
    return config, torch_model, device, backend


def _chunk(config, tail):
    """One full chunk: real signal up front, `tail` behaviour after it."""
    plan = ChunkingPlan.from_config(config)
    rng = np.random.default_rng(SEED)
    chunk = (rng.standard_normal((2, plan.chunk_size)) * 0.1).astype(np.float32)
    if tail == "zeros":
        chunk[:, SIGNAL_SAMPLES:] = 0.0
    elif tail == "near_silent":
        chunk[:, SIGNAL_SAMPLES:] *= 1e-6
    return chunk


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_including_silence(models, tail):
    import mlx.core as mx

    config, torch_model, device, backend = models
    chunk = _chunk(config, tail)

    with torch.no_grad():
        reference = torch_model(torch.from_numpy(chunk).unsqueeze(0).to(device))[0]
    reference = reference.cpu().numpy()

    candidate = backend.model(mx.array(chunk)[None])[0]
    mx.eval(candidate)
    candidate = np.array(candidate)

    assert reference.shape == candidate.shape
    worst = float(np.abs(reference - candidate).max())
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: Torch-vs-MLX max abs {worst:.3e} exceeds "
        f"{MAX_ABS_TOLERANCE:.0e}. If this fired only for a silent tail, suspect "
        f"exact_zero_safe_rfft in bs_roformer/mlx/model.py -- investigate rather "
        f"than widen the tolerance"
    )


VARIANT_MODELS = {
    "large_inst": "roformer-model-bs-roformer-large-inst-by-pcunwa",
    "fno": "roformer-model-bs-roformer-fno-instrumental-by-pcunwa",
    "hyperace": "roformer-model-bs-roformer-hyperace-v2-instrumental-by-pcunwa",
}
VARIANT_TOLERANCE = 1e-5


@pytest.mark.parametrize("variation,slug", sorted(VARIANT_MODELS.items()))
def test_variant_heads_match_torch_end_to_end(tmp_path, variation, slug):
    """Each ported mask-estimator head, through the real public path.

    These are the four registry checkpoints MLX could not run at all until their
    heads were written, and the reason they could not is worth remembering: the
    MLX model swallows unknown constructor arguments, so before this it built a
    plain MaskEstimator without raising and computed the wrong thing.

    Recorded worst-case max abs against Torch on MPS (M2, 5 s seeded stereo):
    large_inst 5.364e-07, fno 3.874e-07, hyperace 6.258e-07.
    """
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'bs-roformer-infer[mlx]'")
    if not mps_available():
        pytest.skip("needs an Apple Silicon Mac and an arm64 torch build")
    try:
        ensure_model_assets(slug, download_missing=False)
    except (FileNotFoundError, KeyError):
        pytest.skip(f"{slug} is not cached; run bs-roformer-download --model {slug}")

    import soundfile as sf

    from bs_roformer import BSRoformerSession

    inputs = tmp_path / "in"
    inputs.mkdir()
    rng = np.random.default_rng(SEED)
    sf.write(
        inputs / "probe.wav",
        (rng.standard_normal((5 * 44100, 2)) * 0.1).astype(np.float32),
        44100,
        subtype="FLOAT",
    )

    produced = {}
    for backend, device in (("torch", "mps"), ("mlx", None)):
        with BSRoformerSession(model_name=slug, backend=backend, device=device) as session:
            manifest = session.infer(inputs, store_dir=tmp_path / backend)
        produced[backend] = {
            output.output_id: sf.read(output.output_path)[0] for output in manifest.outputs
        }

    assert set(produced["torch"]) == set(produced["mlx"])
    for stem, reference in produced["torch"].items():
        worst = float(np.abs(reference - produced["mlx"][stem]).max())
        assert worst < VARIANT_TOLERANCE, (
            f"{variation}/{stem}: Torch-vs-MLX max abs {worst:.3e} exceeds "
            f"{VARIANT_TOLERANCE:.0e}; investigate the head or its weight mapping "
            f"rather than widening the tolerance"
        )


class _DeviceRequest:
    def __init__(self, device):
        self.device = device
