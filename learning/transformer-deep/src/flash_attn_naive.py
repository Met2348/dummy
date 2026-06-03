"""FlashAttention v1 — naive Python 教学实现 (无 Triton).

教学目标：
    1. 演示 tiling + online softmax
    2. 数值上与 vanilla attention 等价
    3. 不要求 GPU (CPU 跑得通)

运行：
    python flash_attn_naive.py
"""
from __future__ import annotations

import math

import torch


def vanilla_attn(q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """标准 attention 作为 ground-truth."""
    b, h, t, d = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if causal:
        mask = torch.full((t, t), float("-inf"), device=q.device,
                          dtype=q.dtype).triu_(1)
        scores = scores + mask
    attn = scores.softmax(dim=-1)
    return attn @ v


def flash_attn_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     block_r: int = 32, block_c: int = 32,
                     causal: bool = False) -> torch.Tensor:
    """Tiled flash-attention with online softmax.

    q, k, v: (b, h, t, d)
    返回 O 同 shape。
    """
    b, h, t, d = q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(q)
    l = torch.zeros(b, h, t, 1, device=q.device, dtype=q.dtype)
    m = torch.full((b, h, t, 1), float("-inf"), device=q.device, dtype=q.dtype)

    for i_start in range(0, t, block_r):
        i_end = min(i_start + block_r, t)
        Qi = q[:, :, i_start:i_end]                       # (b, h, B_r, d)
        Oi = O[:, :, i_start:i_end].clone()
        li = l[:, :, i_start:i_end].clone()
        mi = m[:, :, i_start:i_end].clone()

        for j_start in range(0, t, block_c):
            j_end = min(j_start + block_c, t)
            # causal: 完全 j > i 块跳过
            if causal and j_start > i_end - 1:
                continue
            Kj = k[:, :, j_start:j_end]
            Vj = v[:, :, j_start:j_end]
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale     # (b, h, B_r, B_c)
            if causal:
                # 对角线块需要 mask
                pos_i = torch.arange(i_start, i_end, device=q.device).view(-1, 1)
                pos_j = torch.arange(j_start, j_end, device=q.device).view(1, -1)
                Sij = Sij.masked_fill(pos_j > pos_i, float("-inf"))
            # online softmax 更新
            m_new = torch.maximum(mi, Sij.amax(dim=-1, keepdim=True))
            p = torch.exp(Sij - m_new)
            scale_old = torch.exp(mi - m_new)
            li = scale_old * li + p.sum(dim=-1, keepdim=True)
            Oi = scale_old * Oi + p @ Vj
            mi = m_new

        O[:, :, i_start:i_end] = Oi / li
        l[:, :, i_start:i_end] = li
        m[:, :, i_start:i_end] = mi

    return O


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out_vanilla = vanilla_attn(q, k, v, causal=True)
    out_flash = flash_attn_naive(q, k, v, block_r=4, block_c=4, causal=True)
    diff = (out_vanilla - out_flash).abs().max().item()
    print(f"vanilla vs flash naive max diff: {diff:.2e}")
    assert diff < 1e-4, f"FA naive 数值不一致: {diff}"
    print("✓ Online softmax + tiling 等价于 vanilla")
