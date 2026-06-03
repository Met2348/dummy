"""Dr. GRPO (Sea AI 2025.03) — 修复 GRPO length bias.

GRPO 标准 advantage normalize 用 std:
    A = (R - mean) / std
但 std 在 reward 集中时被 outlier 主导，长 response 被过度奖励.

Dr. GRPO:
    1. 用 mean absolute deviation (MAD) 替代 std
    2. 加 length penalty: A_adj = A - β·log|y|
"""
from __future__ import annotations

import torch


def mad_normalize(rewards: torch.Tensor, k: int, eps: float = 1e-6) -> torch.Tensor:
    """MAD-based group advantage."""
    R = rewards.reshape(-1, k)
    mean = R.mean(dim=1, keepdim=True)
    mad = (R - mean).abs().mean(dim=1, keepdim=True) + eps
    A = (R - mean) / mad
    return A.reshape(-1)


def dr_grpo_advantage(rewards: torch.Tensor,
                      response_lens: torch.Tensor,
                      k: int,
                      beta_len: float = 0.05) -> torch.Tensor:
    """Dr. GRPO advantage = MAD group advantage − β·log|y|."""
    A = mad_normalize(rewards, k)
    length_penalty = beta_len * response_lens.float().clamp(min=1).log()
    return A - length_penalty


def compare_grpo_vs_dr():
    """对照: outlier 存在时 std-normalize 与 MAD-normalize 差异."""
    print("GRPO vs Dr.GRPO advantage (with outlier)")
    rewards = torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 100.0])  # outlier
    k = 8
    # GRPO
    mean = rewards.mean(); std = rewards.std() + 1e-8
    A_g = (rewards - mean) / std
    # Dr.GRPO
    A_dr = mad_normalize(rewards, k)
    print(f"  rewards: {rewards.tolist()}")
    print(f"  GRPO  A: {[round(a, 3) for a in A_g.tolist()]}")
    print(f"  Dr.GRPO A: {[round(a, 3) for a in A_dr.tolist()]}")
    print("  → Dr.GRPO 对 outlier 不敏感，正常样本的 advantage 不被压扁")


if __name__ == "__main__":
    print("Dr. GRPO minimal\n" + "=" * 50)
    compare_grpo_vs_dr()
