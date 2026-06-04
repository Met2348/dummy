"""YaRN — NTK by parts + attention temperature (Peng 2023.09)."""
from __future__ import annotations

import math

import torch


def _yarn_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    """ramp function: 1 below low, 0 above high, linear in between."""
    linear_ramp = torch.linspace(0, 1, dim)
    ramp = (linear_ramp - low) / max(high - low, 1e-6)
    return torch.clamp(1 - ramp, 0, 1)


def yarn_cos_sin(t: int, dim: int, base: float = 10000.0,
                 factor: float = 4.0, original_max_pos: int = 2048,
                 device=None, dtype=torch.float32):
    """YaRN-style scaling.

    factor:        scaling factor (e.g. 4 for 8k → 32k)
    original_max:  原训练 max position
    """
    half_d = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
                                             device=device).float() / dim))
    # 1) NTK-aware: scale base for low freq
    new_base = base * (factor ** (dim / max(dim - 2, 1)))
    inv_freq_ntk = 1.0 / (new_base ** (torch.arange(0, dim, 2,
                                                     device=device).float() / dim))

    # 2) ramp: 高频用原 inv_freq, 低频用 NTK
    # high freq → small dim index; low freq → high dim index
    pi_freq = inv_freq / factor       # full PI
    mask = _yarn_ramp(low=0.5, high=0.9, dim=half_d).to(device or "cpu")
    inv_freq_yarn = mask * inv_freq + (1 - mask) * pi_freq

    pos = torch.arange(t, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq_yarn[None, :]
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    # 3) attention temperature (返回作为额外因子)
    attn_scale = math.sqrt(1.0 / (0.1 * math.log(factor) + 1.0))
    return cos, sin, attn_scale


if __name__ == "__main__":
    cos, sin, scale = yarn_cos_sin(t=32, dim=16, factor=4.0,
                                    original_max_pos=8)
    print(f"YaRN cos {cos.shape}, attn_scale {scale:.4f}")
