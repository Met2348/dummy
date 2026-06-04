"""Position Interpolation (Meta 2023)."""
from __future__ import annotations

import torch

from common import inv_freq as base_inv_freq


def pi_cos_sin(t: int, dim: int, base: float = 10000.0,
               scale_factor: float = 4.0, device=None,
               dtype=torch.float32):
    """PI: 把 position 压缩 scale_factor 倍."""
    inv_freq = base_inv_freq(dim, base, device=device)
    pos = torch.arange(t, device=device, dtype=torch.float32) / scale_factor
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


if __name__ == "__main__":
    cos, sin = pi_cos_sin(t=8, dim=16, scale_factor=4.0)
    print(f"PI cos {cos.shape}")
    print(f"pos 0:  cos[0,0]={cos[0,0]:.4f}  (= 1.0?)")
    # PI 缩小后 pos 4 实际对应原 pos 1
    print(f"pos 4:  cos[4,0]={cos[4,0]:.4f}")
