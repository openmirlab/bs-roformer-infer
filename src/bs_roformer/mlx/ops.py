"""MLX-native tensor reshape primitives and small config/existence helpers.

`pack`/`unpack`/`rearrange` are a minimal einops replacement: MLX has no einops
integration, so the trunk and attention modules need a handful of named reshape
patterns (not a general parser) to move between `(b, f, t, d)`-style layouts.
`batched_group_linear` is the batched-matmul equivalent of running the same
per-band `nn.Linear` in a Python loop, used by the "grouped" fast paths in
`bands.py`. `exists`/`default`/`env_enabled` are the tiny helpers those and the
rest of the MLX trunk lean on for optional values and boolean env flags.

Reads: mlx.core, numpy
"""

import os
from typing import List, Tuple  # noqa: UP035

import mlx.core as mx
import numpy as np


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def env_enabled(name: str, default_value: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default_value)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def batched_group_linear(x_group: mx.array, weights: mx.array, biases: mx.array | None = None) -> mx.array:
    """Apply per-group linear layers in a single batched matmul.

    Args:
        x_group: Shape (B, T, G, In)
        weights: Shape (G, Out, In)
        biases: Optional shape (G, Out)
    """
    out = mx.einsum("btgi,goi->btgo", x_group, weights)
    if biases is not None:
        out = out + biases[None, None, :, :]
    return out


# Use MLX's built-in nn.Sequential, which stores layers in self.layers list
# Weight keys need to use "layers.0", "layers.1", etc. to access them


# MLX-native replacements for einops operations
def pack(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """
    Pack tensors by flattening dimensions according to pattern.

    Patterns:
    - "b * d" : keep first and last dim, flatten middle dims
    - "* t d" : flatten leading dims, keep last two dims
    - "* f d" : flatten leading dims, keep last two dims
    """
    if len(tensors) == 1:
        x = tensors[0]
        original_shape = x.shape

        if pattern == "b * d":
            # Keep first and last dim, flatten middle
            # (b, f, t, d) -> (b, f*t, d)
            if len(x.shape) == 4:
                b, f, t, d = x.shape
                x = mx.reshape(x, (b, f * t, d))
            elif len(x.shape) == 3:
                # Already in the right shape
                pass
            return x, [original_shape]

        elif pattern == "* t d" or pattern == "* f d":
            # Flatten leading dims, keep last two
            # (b, f, t, d) -> (b*f, t, d)
            if len(x.shape) >= 3:
                *leading, t, d = x.shape
                leading_size = int(np.prod(leading))
                x = mx.reshape(x, (leading_size, t, d))
            return x, [original_shape]

        else:
            raise NotImplementedError(f"Pack pattern not implemented: {pattern}")
    else:
        # Stack multiple tensors
        packed = mx.stack(tensors, axis=0)
        shapes = [t.shape for t in tensors]
        return packed, shapes


def unpack(tensor: mx.array, shapes: List, pattern: str) -> List[mx.array]:  # noqa: UP006
    """
    Unpack tensor by restoring original shape.
    """
    if len(shapes) == 1:
        original_shape = shapes[0]
        x = mx.reshape(tensor, original_shape)
        return [x]
    else:
        # Unstack multiple tensors
        return [tensor[i] for i in range(len(shapes))]


def pack_one(tensors: List[mx.array], pattern: str) -> Tuple[mx.array, List]:  # noqa: UP006
    """Pack single tensor - alias for pack."""
    return pack(tensors, pattern)


def unpack_one(tensor: mx.array, shapes: List, pattern: str) -> mx.array:  # noqa: UP006
    """Unpack single tensor - returns first element."""
    return unpack(tensor, shapes, pattern)[0]


def rearrange(x: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    MLX-native implementation of einops rearrange for BS-Roformer patterns.
    Handles the specific rearrange patterns used in this model.
    """
    if "->" not in pattern:
        raise ValueError(f"Invalid pattern: {pattern}")

    input_pattern, output_pattern = pattern.split("->")
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()

    # Pattern: "b n (qkv h d) -> qkv b h n d" with qkv=3, h=heads
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h n d":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 1, 4))  # qkv, b, h, n, d
        return x

    # Pattern: "b n (qkv h d) -> qkv b h d n" with qkv=3, h=heads (for freq transformer)
    if input_pattern == "b n (qkv h d)" and output_pattern == "qkv b h d n":
        b, n, _ = x.shape
        qkv = axes_lengths['qkv']
        h = axes_lengths['h']
        d = x.shape[-1] // (qkv * h)
        x = mx.reshape(x, (b, n, qkv, h, d))
        x = mx.transpose(x, (2, 0, 3, 4, 1))  # qkv, b, h, d, n
        return x

    # Pattern: "b h n d -> b n h d"
    if input_pattern == "b h n d" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h d n -> b n h d"
    if input_pattern == "b h d n" and output_pattern == "b n h d":
        return mx.transpose(x, (0, 3, 1, 2))

    # Pattern: "b n h d -> b h n d"
    if input_pattern == "b n h d" and output_pattern == "b h n d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b h n d -> b n (h d)"
    if input_pattern == "b h n d" and output_pattern == "b n (h d)":
        b, h, n, d = x.shape
        return mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, n, h * d))

    # Pattern: "b h d n -> b n (h d)"
    if input_pattern == "b h d n" and output_pattern == "b n (h d)":
        b, h, d, n = x.shape
        return mx.reshape(mx.transpose(x, (0, 3, 1, 2)), (b, n, h * d))

    # Pattern: "b n h -> b h n 1"
    if input_pattern == "b n h" and output_pattern == "b h n 1":
        return mx.transpose(x, (0, 2, 1))[..., None]

    # Pattern: "b c t -> (b c) t"
    if input_pattern == "b c t" and output_pattern == "(b c) t":
        b, c, t = x.shape
        return mx.reshape(x, (b * c, t))

    # Pattern: "(b c) f t complex -> b (f c) t complex" with c=channels
    if input_pattern == "(b c) f t complex" and output_pattern == "b (f c) t complex":
        c = axes_lengths['c']
        bc, f, t, complex_dim = x.shape
        b = bc // c
        x = mx.reshape(x, (b, c, f, t, complex_dim))
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # b, f, c, t, complex
        x = mx.reshape(x, (b, f * c, t, complex_dim))
        return x

    # Pattern: "b n (f c) t -> (b n c) f t" with c=channels
    if input_pattern == "b n (f c) t" and output_pattern == "(b n c) f t":
        c = axes_lengths['c']
        b, n, fc, t = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, f, c, t))
        x = mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, c, f, t
        x = mx.reshape(x, (b * n * c, f, t))
        return x

    # Pattern: "(b n c) t -> b n c t" with b=batch, n=stems, c=channels
    if input_pattern == "(b n c) t" and output_pattern == "b n c t":
        b = axes_lengths['b']
        n = axes_lengths['n']
        c = axes_lengths['c']
        _bnc, t = x.shape
        return mx.reshape(x, (b, n, c, t))

    # Pattern: "b 1 c t -> b c t"
    if input_pattern == "b 1 c t" and output_pattern == "b c t":
        return mx.squeeze(x, axis=1)

    # Pattern: "b f t c -> b t (f c)"
    if input_pattern == "b f t c" and output_pattern == "b t (f c)":
        b, f, t, c = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # b, t, f, c
        return mx.reshape(x, (b, t, f * c))

    # Pattern: "b t f d -> b f t d"
    if input_pattern == "b t f d" and output_pattern == "b f t d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b f t d -> b t f d"
    if input_pattern == "b f t d" and output_pattern == "b t f d":
        return mx.transpose(x, (0, 2, 1, 3))

    # Pattern: "b n t (f c) -> b n f t c" with c=2
    if input_pattern == "b n t (f c)" and output_pattern == "b n f t c":
        c = axes_lengths['c']
        b, n, t, fc = x.shape
        f = fc // c
        x = mx.reshape(x, (b, n, t, f, c))
        return mx.transpose(x, (0, 1, 3, 2, 4))  # b, n, f, t, c

    raise NotImplementedError(f"Rearrange pattern not implemented: {pattern}")
