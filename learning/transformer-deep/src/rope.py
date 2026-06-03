"""RoPE — Rotary Position Embedding (Su 2021).

教学目标：
    1. 复数旋转的实数等价实现
    2. interleaved (RoPE 原版) vs split (HF 早期)
    3. 与 flash-attn 接口对齐

约定：
    本实现采用 **interleaved** 配对 (dim 2i 与 dim 2i+1 配对)，
    与 RoPE 原论文 / flash-attn 一致。
    HF Llama 用 split (前 d/2 与后 d/2 配对) → 不兼容！

运行：
    python rope.py
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def build_cos_sin(t: int, dim: int, base: float = 10000.0,
                  device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，shape (t, dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                            dtype=torch.float32) / dim))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]                # (t, dim/2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor,
                           sin: torch.Tensor) -> torch.Tensor:
    """interleaved RoPE — dim 2i/2i+1 配对.

    x:   (..., t, dim)
    cos: (t, dim/2)
    sin: (t, dim/2)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


def apply_rope_split(x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """HF 风格 split — 前 d/2 与后 d/2 配对."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)


class RoPE(nn.Module):
    """RoPE module 工程接口."""

    def __init__(self, dim: int, base: float = 10000.0,
                 interleaved: bool = True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, t: int, device, dtype):
        pos = torch.arange(t, device=device, dtype=torch.float32)
        angles = pos[:, None] * self.inv_freq.to(device)[None, :]
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k: (..., t, dim).  返回 rotated (q, k).

        pos_offset: 推理时 KV cache 起始位置.
        """
        t = q.shape[-2]
        cos, sin = self._cos_sin(t + pos_offset, q.device, q.dtype)
        cos = cos[pos_offset:pos_offset + t]
        sin = sin[pos_offset:pos_offset + t]
        # broadcast 到 q 的多余 batch/head 维
        # cos: (t, d/2)  →  (1, 1, t, d/2)
        for _ in range(q.dim() - 2):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if self.interleaved:
            return (apply_rope_interleaved(q, cos, sin),
                    apply_rope_interleaved(k, cos, sin))
        return (apply_rope_split(q, cos, sin),
                apply_rope_split(k, cos, sin))


if __name__ == "__main__":
    torch.manual_seed(0)
    rope = RoPE(dim=16, base=10000.0)
    q = torch.randn(1, 2, 8, 16)  # (b, h, t, d)
    k = torch.randn(1, 2, 8, 16)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape
    print("✓ shape preserved", qr.shape)

    # 相对位置不变性 sanity check
    q0 = torch.ones(1, 1, 1, 16)
    k0 = torch.ones(1, 1, 1, 16)
    # m=2, n=5  vs  m=10, n=13 ：相对位置都是 3
    def score(m, n):
        cos_q, sin_q = build_cos_sin(m + 1, 16)
        cos_k, sin_k = build_cos_sin(n + 1, 16)
        qr = apply_rope_interleaved(q0, cos_q[m:m+1], sin_q[m:m+1])
        kr = apply_rope_interleaved(k0, cos_k[n:n+1], sin_k[n:n+1])
        return (qr * kr).sum().item()
    s1 = score(2, 5)
    s2 = score(10, 13)
    print(f"score m=2,n=5:  {s1:.4f}")
    print(f"score m=10,n=13: {s2:.4f}")
    print(f"relative-only? |diff|={abs(s1-s2):.6f}  (should be ~0)")
