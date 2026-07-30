"""PyTorch -> MLX weight conversion for BSRoformerMLX, plus a strict load gate.

`convert_torch_to_mlx_weights` (vendored verbatim, see below) renames and
restructures a PyTorch `state_dict`'s keys to match the MLX module tree built
by `model.py` -- gamma->weight, `layers.N.K` block indices, Sequential
`layers.N` insertion, band-split/mask-estimator submodule naming. Upstream
then loads the result with `model.load_weights(list(weights.items()),
strict=False)`, which **silently discards** any key that doesn't match the
module tree: a checkpoint can "load" with whole layers left at random
initialization and produce plausible-looking garbage with no error. This
module adds `load_converted_weights()`, which diffs the model's own parameter
keys (via `mlx.utils.tree_flatten`) against the converted weight keys and
raises a `ValueError` naming the mismatch *before* calling `load_weights` --
callers should use this instead of calling `load_weights` directly.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/separator/models/roformer/loader.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.
    (`convert_torch_to_mlx_weights` and `_to_numpy` are the vendored functions;
    `load_converted_weights` below is new, not from upstream.)

Reads: mlx.core, mlx.nn, mlx.utils (tree_flatten), numpy
"""

import logging
import re
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


def _apply_sequential_rules(mlx_key: str) -> str:
    """MLX Sequential wraps its children in `.layers`; Torch's does not."""
    mlx_key = re.sub(r"to_freqs\.(\d+)\.0\.", r"to_freqs_\1.", mlx_key)
    mlx_key = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", mlx_key)
    mlx_key = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", mlx_key)
    mlx_key = re.sub(r"(to_freqs_\d+)\.(\d+)\.", r"\1.layers.\2.", mlx_key)
    return mlx_key


#: torch `mask_estimators.N.layers.I.{0|1}.layers.0.{0|1}.<rest>` -- Large-Inst's
#: four time/freq Transformer pairs. Both Torch nesting levels collapse into one
#: hop because the MLX head uses a bare depth-1 layer, not a depth-loop wrapper.
_LARGE_INST_BLOCK = re.compile(
    r"^mask_estimators\.(\d+)\.layers\.(\d+)\.([01])\.layers\.0\.([01])\.(.+)$"
)
_AXIS = {"0": "time_transformer", "1": "freq_transformer"}
_SUBMODULE = {"0": "attn", "1": "ff"}


def _convert_head_key(key: str, weight: np.ndarray, variant: str | None):
    """Rename and reshape one variant-head tensor, or return None to pass it on.

    Returns None for anything that is not a variant head's parameter, so the trunk
    path stays exactly as verified for the stock model.
    """
    if variant in (None, "", "mlp") or not key.startswith("mask_estimators."):
        return None

    if variant == "large_inst":
        block = _LARGE_INST_BLOCK.match(key)
        if block:
            estimator, layer, axis, submodule, rest = block.groups()
            prefix = f"mask_estimators_{estimator}.layers_{layer}.{_AXIS[axis]}"
            if rest == "rotary_embed.freqs":
                # NOT skipped, unlike the trunk's. This checkpoint's rotary
                # frequencies drifted during training -- they differ from the
                # theta=10000 default by up to 0.0088 and go negative in places,
                # which no fixed formula produces. Dropping them looked correct
                # (every key matched) and computed the wrong thing. Loaded raw;
                # the head inverts them, because mx.fast.rope wants the reciprocal
                # of what rotary_embedding_torch stores.
                return f"{prefix}.attn.rope_freqs", weight
            # `.gamma` -> `.weight` applies here too: taking the early return
            # without it left 16 norm tensors unmatched, which upstream's
            # strict=False would have loaded as silence-shaped garbage.
            rest = rest.replace(".gamma", ".weight")
            return f"{prefix}.{_SUBMODULE[submodule]}.{rest}", weight

    if "rotary_embed.freqs" in key:
        return None

    mlx_key = re.sub(r"^mask_estimators\.(\d+)", r"mask_estimators_\1", key)

    if variant == "fno":
        # Pure renames: the head keeps Torch's channels-first layout and its
        # 1x1 convolutions keep their native (out, in, 1) weight shape, so no
        # tensor is reshaped here.
        mlx_key = re.sub(
            r"\.(convs|fno_skips|channel_mlp|channel_mlp_skips)\.(\d+)\.",
            r".\1_\2.",
            mlx_key,
        )
        mlx_key = re.sub(r"\.fcs\.(\d+)\.", r".fcs_\1.", mlx_key)
        mlx_key = mlx_key.replace(".weight.tensor", ".weight")
        mlx_key = re.sub(r"(fno_skips_\d+)\.conv\.weight", r"\1.weight", mlx_key)
        return mlx_key, weight

    if variant == "hyperace":
        if key.endswith(".gamma"):
            # GatedFusion stores its scale as (1, C, 1, 1) to broadcast over
            # NCHW. MLX is NHWC, so it needs a plain (C,) to broadcast over the
            # last axis. Only singleton dims are dropped; values are untouched.
            return mlx_key.replace(".gamma", ".weight"), weight.reshape(-1)
        if mlx_key.endswith(".weight") and weight.ndim == 4:
            # Conv2d: Torch OIHW, MLX OHWI. Nothing else in this head is rank-4
            # (no Linear is, and the norms are rank-1), so the rank is a safe test.
            return mlx_key, weight.transpose(0, 2, 3, 1)

    return mlx_key.replace(".gamma", ".weight"), weight


def convert_torch_to_mlx_weights(
    state_dict: dict[str, Any], *, variant: str | None = None
) -> dict[str, mx.array]:
    """Convert PyTorch state dict to MLX weights format.

    Handles:
    1. Parameter name mapping (gamma -> weight for norms)
    2. Module path restructuring for MLX module tree
    3. Sequential layer indexing (module.N -> module.layers.N)
    4. Variant mask-estimator heads, which own their own naming (`variant`)
    """
    # Detect if model has linear transformers
    has_linear_transformers = any(
        re.match(r"^layers\.\d+\.2\.", key) for key in state_dict.keys()  # noqa: SIM118
    )
    logger.info(f"Checkpoint has linear transformers: {has_linear_transformers}")

    mlx_weights = {}

    for key, value in state_dict.items():
        # Variant mask-estimator heads own their own naming. They are converted
        # first and then skip the trunk's regexes entirely: the generic
        # transformer rule below misreads Large-Inst's outer time/freq selector as
        # an attn/ff selector, which produced a key set that matched perfectly and
        # computed the wrong thing.
        head = _convert_head_key(key, _to_numpy(value), variant)
        if head is not None:
            head_key, head_weight = head
            mlx_weights[_apply_sequential_rules(head_key)] = mx.array(head_weight)
            continue

        # Skip rotary embedding buffers
        if "rotary_embed.freqs" in key:
            continue

        # Convert to numpy
        numpy_weight = _to_numpy(value)

        # Map parameter names
        mlx_key = key.replace(".gamma", ".weight")

        # Restructure band_split paths
        if "band_split.to_features." in mlx_key:
            parts = mlx_key.split(".")
            if len(parts) >= 5 and parts[2].isdigit():
                band_idx = parts[2]
                submodule_idx = parts[3]
                param_name = ".".join(parts[4:])
                if submodule_idx == "0":
                    mlx_key = f"band_split.to_features_{band_idx}.norm.{param_name}"
                elif submodule_idx == "1":
                    mlx_key = f"band_split.to_features_{band_idx}.linear.{param_name}"

        # Restructure main block transformer paths
        main_block_match = re.match(r"^layers\.(\d+)\.([012])\.(.+)$", mlx_key)
        if main_block_match:
            block_idx = main_block_match.group(1)
            transformer_idx = main_block_match.group(2)
            rest = main_block_match.group(3)

            if has_linear_transformers:
                names = ["linear_transformer", "time_transformer", "freq_transformer"]
                mlx_key = f"layers_{block_idx}.{names[int(transformer_idx)]}.{rest}"
            else:
                if transformer_idx == "0":
                    mlx_key = f"layers_{block_idx}.time_transformer.{rest}"
                elif transformer_idx == "1":
                    mlx_key = f"layers_{block_idx}.freq_transformer.{rest}"

        # Restructure individual transformer layer paths
        transformer_match = re.search(r"(\.layers)\.(\d+)\.([01])\.", mlx_key)
        if transformer_match:
            prefix = mlx_key[: transformer_match.start()]
            layer_idx = transformer_match.group(2)
            submodule = transformer_match.group(3)
            suffix = mlx_key[transformer_match.end() :]

            if submodule == "0":
                mlx_key = f"{prefix}.layers_{layer_idx}.attn.{suffix}"
            elif submodule == "1":
                mlx_key = f"{prefix}.layers_{layer_idx}.ff.{suffix}"

        # Restructure mask_estimators
        if "mask_estimators." in mlx_key:
            mlx_key = re.sub(r"mask_estimators\.(\d+)", r"mask_estimators_\1", mlx_key)
            mlx_key = re.sub(r"to_freqs\.(\d+)\.0\.", r"to_freqs_\1.", mlx_key)

        # MLX Sequential: insert "layers." before numeric indices
        mlx_key = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", mlx_key)
        mlx_key = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", mlx_key)
        mlx_key = re.sub(r"(to_freqs_\d+)\.(\d+)\.", r"\1.layers.\2.", mlx_key)

        mlx_weights[mlx_key] = mx.array(numpy_weight)

    logger.debug(f"Converted {len(mlx_weights)} tensors from PyTorch to MLX format")
    return mlx_weights


def _to_numpy(value) -> np.ndarray:
    """Convert a weight value to numpy array, handling torch tensors."""
    try:
        # Try torch tensor first
        return value.cpu().numpy()
    except AttributeError:
        return np.array(value)


def load_converted_weights(model: nn.Module, mlx_weights: dict[str, mx.array]) -> None:
    """Load `mlx_weights` into `model`, refusing a silent partial load.

    `model.load_weights(..., strict=False)` on its own accepts any degree of
    mismatch between the checkpoint and the module tree, dropping whatever
    doesn't line up without a warning. This checks first: every one of the
    model's own parameter keys (from `mlx.utils.tree_flatten(model.parameters())`)
    must be present in `mlx_weights`, and every key in `mlx_weights` must be
    consumed by the model -- otherwise a `ValueError` is raised naming counts
    and up to 5 example keys on each side, so a conversion bug or a mismatched
    checkpoint fails loudly instead of loading a partially-random model.
    """
    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(mlx_weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            example = ", ".join(unmatched_model[:5])
            parts.append(
                f"{len(unmatched_model)} model parameters unmatched (e.g. {example})"
            )
        if dropped_weights:
            example = ", ".join(dropped_weights[:5])
            parts.append(
                f"{len(dropped_weights)} converted tensors dropped (e.g. {example})"
            )
        raise ValueError("MLX weight conversion incomplete: " + ", ".join(parts))

    model.load_weights(list(mlx_weights.items()), strict=False)
