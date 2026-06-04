"""Ring Attention naive — 单 GPU 模拟多 GPU 通信."""
from __future__ import annotations

import math

import torch


def vanilla_attn(q, k, v):
    d = q.shape[-1]
    return ((q @ k.transpose(-2, -1)) / math.sqrt(d)).softmax(dim=-1) @ v


def ring_attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          n_rank: int = 4) -> torch.Tensor:
    """单 GPU 模拟 ring attention.

    把 q, k, v 切成 n_rank 块沿 seq 维度。
    """
    b, h, t, d = q.shape
    assert t % n_rank == 0, "seq must divisible by n_rank"
    chunk = t // n_rank
    out = torch.zeros_like(q)
    for rank in range(n_rank):
        q_rank = q[..., rank * chunk: (rank + 1) * chunk, :]
        # 用 online softmax 累积所有 K_j, V_j
        m_run = torch.full((b, h, chunk, 1), float("-inf"),
                           device=q.device, dtype=q.dtype)
        l_run = torch.zeros(b, h, chunk, 1, device=q.device, dtype=q.dtype)
        O_run = torch.zeros(b, h, chunk, d, device=q.device, dtype=q.dtype)
        for r in range(n_rank):
            # 模拟 ring: K, V 来自 (rank+r) mod n_rank
            j = (rank + r) % n_rank
            k_j = k[..., j * chunk: (j + 1) * chunk, :]
            v_j = v[..., j * chunk: (j + 1) * chunk, :]
            scores = (q_rank @ k_j.transpose(-2, -1)) / math.sqrt(d)
            m_new = torch.maximum(m_run, scores.amax(-1, keepdim=True))
            p = torch.exp(scores - m_new)
            scale_old = torch.exp(m_run - m_new)
            l_run = scale_old * l_run + p.sum(-1, keepdim=True)
            O_run = scale_old * O_run + p @ v_j
            m_run = m_new
        out[..., rank * chunk: (rank + 1) * chunk, :] = O_run / l_run
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out_vanilla = vanilla_attn(q, k, v)
    out_ring = ring_attention_naive(q, k, v, n_rank=4)
    diff = (out_vanilla - out_ring).abs().max().item()
    print(f"vanilla vs ring naive max diff: {diff:.2e}")
    assert diff < 1e-4
    print("✓ Ring attention 数值等价")
