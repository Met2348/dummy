"""长上下文共用工具 — RoPE freq helpers."""
from __future__ import annotations

import math

import torch


def inv_freq(dim: int, base: float = 10000.0,
             device=None, dtype=torch.float32) -> torch.Tensor:
    """RoPE inverse frequencies, shape (dim/2,)."""
    return 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                        dtype=dtype) / dim))


def build_cos_sin(t: int, inv_freq_: torch.Tensor,
                  device=None, dtype=torch.float32):
    pos = torch.arange(t, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq_[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor,
                            sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out
