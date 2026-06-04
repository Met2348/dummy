"""NTK-aware RoPE scaling (LocalLlama 2023.06)."""
from __future__ import annotations

import torch


def ntk_cos_sin(t: int, dim: int, base: float = 10000.0,
                scale_factor: float = 4.0, device=None,
                dtype=torch.float32):
    """NTK-aware: 改 base 而非 position."""
    # NTK formula: base_new = base * scale^(d/(d-2))
    new_base = base * (scale_factor ** (dim / max(dim - 2, 1)))
    inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


if __name__ == "__main__":
    cos, sin = ntk_cos_sin(t=16, dim=16, base=10000, scale_factor=4)
    print(f"NTK cos {cos.shape}")
    # 比较与原 RoPE 的角度差
    from common import inv_freq, build_cos_sin
    orig_cos, _ = build_cos_sin(t=16, inv_freq_=inv_freq(16, 10000))
    diff = (cos - orig_cos).abs().mean().item()
    print(f"angle diff vs base RoPE: {diff:.4f}")
