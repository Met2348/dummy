"""MoE 共用工具 — 负载统计 / 容量计算 / expert init."""
from __future__ import annotations

import torch


def expert_load(top_k_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """计算每 expert 接到的 token 数, shape (n_experts,)."""
    flat = top_k_idx.flatten()
    counts = torch.bincount(flat, minlength=n_experts).float()
    return counts


def load_balance_loss(gates: torch.Tensor, top_k_idx: torch.Tensor,
                      n_experts: int) -> torch.Tensor:
    """Switch Transformer aux loss: encourage balance.

    gates:      (b*t, n_experts) softmax 后概率
    top_k_idx:  (b*t, top_k) selected experts
    """
    n_tokens = gates.shape[0]
    f = expert_load(top_k_idx, n_experts) / n_tokens  # 实际频率
    p = gates.mean(dim=0)                                # 平均概率
    return n_experts * (f * p).sum()


def capacity(n_tokens: int, n_experts: int, top_k: int,
             factor: float = 1.25) -> int:
    """每 expert 容量上限 (overflow 丢)."""
    return int(factor * n_tokens * top_k / n_experts)


def init_expert(linear: torch.nn.Linear, std: float = 0.02) -> None:
    torch.nn.init.normal_(linear.weight, std=std)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)
