"""Acceptance inventory for the pcunwa BS-RoFormer repositories."""

import pytest

from bs_roformer.checkpoints import load_checkpoints
from bs_roformer.model_registry import MODEL_REGISTRY


PCUNWA_CHECKPOINTS = {
    "bs_roformer_leap_inst.ckpt",
    "bs_roformer_leap_voc.ckpt",
    "bs_leap_xe_inst.ckpt",
    "bs_leap_xe_voc.ckpt",
    "bs_exp_siameseroformer.ckpt",
    "bs_hyperace.ckpt",
    "bs_roformer_inst_hyperacev2.ckpt",
    "bs_roformer_voc_hyperacev2.ckpt",
    "bs_roformer_fno.ckpt",
    "bs_large_v2_inst.ckpt",
    "BS_Inst_EXP_VRL.ckpt",
    "BS-Roformer-Resurrection-Inst.ckpt",
    "BS-Roformer-Resurrection.ckpt",
    "bs_roformer_revive.ckpt",
    "bs_roformer_revive2.ckpt",
    "bs_roformer_revive3e.ckpt",
}


def _checkpoint_artifacts():
    data = load_checkpoints()["models"]
    return {
        artifact["name"]: (slug, model, artifact)
        for slug, model in data.items()
        for artifact in model["artifacts"]
        if artifact["kind"] == "checkpoint"
    }


def test_every_pcunwa_bs_checkpoint_is_registered_once():
    artifacts = _checkpoint_artifacts()
    assert PCUNWA_CHECKPOINTS <= artifacts.keys()
    assert len([name for name in artifacts if name in PCUNWA_CHECKPOINTS]) == 16
    for checkpoint in PCUNWA_CHECKPOINTS:
        slug, model, artifact = artifacts[checkpoint]
        assert slug in {entry.slug for entry in MODEL_REGISTRY.list()}
        assert "huggingface.co/pcunwa/" in artifact["url"]
        assert len(artifact["sha256"]) == 64
        assert artifact["size"] > 0


@pytest.mark.parametrize(
    ("slug", "variation"),
    [
        ("roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa", "mlp"),
        ("roformer-model-bs-roformer-leap-xe-vocals-by-pcunwa", "mlp"),
        ("roformer-model-bs-roformer-siamese-vocals-by-pcunwa", "siamese"),
        ("roformer-model-bs-roformer-hyperace-v1-instrumental-by-pcunwa", "hyperace_v1"),
        ("roformer-model-bs-roformer-value-residual-instrumental-by-pcunwa", "value_residual"),
    ],
)
def test_new_variation_registry_metadata(slug, variation):
    assert MODEL_REGISTRY.get(slug).variation == variation
