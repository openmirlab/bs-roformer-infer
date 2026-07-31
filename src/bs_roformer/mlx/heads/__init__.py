"""Mask-estimator heads for the MLX trunk -- one owner for variant selection.

The BS-RoFormer trunk is shared; what changes between checkpoints is the head that
turns trunk features into a complex mask. Torch selects one in
``bs_roformer._create_mask_estimator``; this is its MLX counterpart, kept as a
registry rather than a chain of ``if`` statements inside the model so adding a head
touches one table.

Heads are imported lazily: a checkpoint that never asks for HyperACE should not pay
for importing it, and a missing head must fail by name rather than at some later
shape mismatch.

Reads: ..bands (MaskEstimator), .large_inst, .fno, .hyperace (all lazily)
"""

from __future__ import annotations

#: Every variation the Torch side's `_create_mask_estimator` knows about. This is
#: the target, not a capability claim -- ask `available_variants()` for that.
VARIANTS = ("mlp", "hyperace", "fno", "large_inst")

#: Which variant lives in which module. "mlp" is built from the trunk itself.
_MODULES = {"hyperace": "hyperace", "fno": "fno", "large_inst": "large_inst"}


def available_variants() -> tuple[str, ...]:
    """The variants whose head module actually exists here.

    Capability must be measured, not declared: a variant listed as supported whose
    module is missing would fail as an ImportError deep in construction instead of
    a clean refusal at backend-selection time. Uses `find_spec`, so asking costs no
    import.
    """
    from importlib.util import find_spec

    found = ["mlp"]
    for variant, module in _MODULES.items():
        if find_spec(f"{__name__}.{module}") is not None:
            found.append(variant)
    return tuple(found)


def build_mask_estimator(
    *,
    variant: str | None,
    dim: int,
    dim_inputs,
    depth: int,
    audio_channels: int,
    mlp_expansion_factor: int,
    use_grouped: bool = False,
):
    """Build the head this checkpoint was trained with, or refuse by name.

    Mirrors the Torch selector's signature so the two cannot drift on which
    arguments a head receives. `use_grouped` is an MLX-only fusion switch that
    applies to the stock head alone.
    """
    variant = variant or "mlp"

    if variant == "mlp":
        from ..bands import MaskEstimator

        return MaskEstimator(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
            use_grouped=use_grouped,
        )
    if variant == "hyperace":
        from .hyperace import HyperACEMaskEstimator

        return HyperACEMaskEstimator(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            audio_channels=audio_channels,
            mlp_expansion_factor=mlp_expansion_factor,
        )
    if variant == "fno":
        from .fno import FNOMaskEstimator

        return FNOMaskEstimator(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
    if variant == "large_inst":
        from .large_inst import LargeInstMaskEstimator

        return LargeInstMaskEstimator(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
    raise ValueError(f"unknown mask_estimator_variant: {variant!r}")


__all__ = ["VARIANTS", "available_variants", "build_mask_estimator"]
