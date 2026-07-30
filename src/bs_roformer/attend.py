"""Attend -- attention core with automatic flash/math/mem-efficient backend selection.

Adapted from lucidrains-style attention wrappers: picks PyTorch's SDPA backend based
on GPU compute capability at construction time (A100 gets flash-only; other CUDA
devices get math/mem-efficient) rather than letting PyTorch guess per call, and falls
back to a plain einsum attention path when `flash=False` so behavior matches the
non-flash checkpoints this package loads. The backend-pinning context manager is
entered only for CUDA tensors, since its flags govern CUDA kernel choice alone and
entering it on CPU/MPS bought nothing but a per-call deprecation warning.
Requires PyTorch >= 2.0 when flash is enabled.

Reads: torch (nn, scaled_dot_product_attention), packaging.version, einops
"""

from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F


# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        is_cuda = q.is_cuda

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        dropout_p = self.dropout if self.training else 0.

        # sdp_kernel's flags only steer CUDA kernel selection, so entering this
        # (deprecated) context manager off CUDA selected nothing while emitting a
        # FutureWarning on every attention call -- dozens per forward on MPS/CPU.
        # Enter it only where it has meaning; elsewhere PyTorch's own dispatch is
        # exactly what the all-enabled cpu_config asked for anyway.
        if not is_cuda or self.cuda_config is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p = dropout_p)

        with torch.backends.cuda.sdp_kernel(**self.cuda_config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p = dropout_p)

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
