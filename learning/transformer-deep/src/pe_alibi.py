"""ALiBi — Attention with Linear Biases (Press et al. 2022)."""
from __future__ import annotations

import torch


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """Geometric sequence m_h = 2^(-8h/H)."""
    start = 2 ** (-8.0 / num_heads)
    return start ** torch.arange(1, num_heads + 1, dtype=torch.float32)


def alibi_bias(t: int, num_heads: int, device=None,
               dtype=torch.float32) -> torch.Tensor:
    """Return bias of shape (num_heads, t, t)."""
    slopes = alibi_slopes(num_heads).to(device=device, dtype=dtype)
    pos = torch.arange(t, device=device, dtype=dtype)
    # diff matrix: bias_{ij} = -|i - j|
    diff = -(pos[None, :] - pos[:, None]).abs()         # (t, t)
    bias = slopes[:, None, None] * diff.unsqueeze(0)     # (h, t, t)
    return bias


if __name__ == "__main__":
    bias = alibi_bias(t=4, num_heads=4)
    print("shape:", bias.shape)
    print("head 0:\n", bias[0])
    print("head 3 (steepest):\n", bias[3])
