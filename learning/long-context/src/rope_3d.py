"""3D RoPE / M-RoPE — Qwen2-VL style."""
from __future__ import annotations

import torch

from common import inv_freq, apply_rope_interleaved


def m_rope(x: torch.Tensor, positions: torch.Tensor,
           base: float = 10000.0) -> torch.Tensor:
    """M-RoPE: x shape (..., dim).

    positions: (..., 3) — (t, h, w) coordinates per token.
    dim 分成 3 段，每段独立 RoPE on对应 axis.
    """
    dim = x.shape[-1]
    seg = dim // 6 * 2     # 每 axis 用 dim/3 (even number)
    if seg == 0:
        seg = 2
    inv_freq_seg = inv_freq(seg, base, device=x.device, dtype=torch.float32)

    pieces = []
    for axis in range(3):
        slice_start = axis * seg
        slice_end = min((axis + 1) * seg, dim)
        if slice_end <= slice_start:
            continue
        x_seg = x[..., slice_start:slice_end]
        pos = positions[..., axis].float()              # (...,)
        angles = pos.unsqueeze(-1) * inv_freq_seg[: (slice_end - slice_start) // 2]
        cos = angles.cos().to(x.dtype)
        sin = angles.sin().to(x.dtype)
        pieces.append(apply_rope_interleaved(x_seg, cos, sin))
    # 剩余维 (dim mod 6)
    tail = x[..., 3 * seg:]
    pieces.append(tail)
    return torch.cat(pieces, dim=-1)


if __name__ == "__main__":
    x = torch.randn(2, 8, 24)
    positions = torch.tensor([[i, 0, 0] for i in range(8)] * 2).view(2, 8, 3)
    y = m_rope(x, positions)
    print(f"M-RoPE out {y.shape}")
