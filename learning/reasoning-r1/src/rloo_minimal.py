"""RLOO (Ahmadian 2024) — Leave-One-Out baseline.

每条 rollout 的 baseline = 其他 k-1 条的均值 (无 critic).
A_i = R_i - mean_{j!=i}(R_j) = (k·R_i - sum R_j) / (k-1)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rloo_advantage(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """A_i = R_i - mean(R_others) within each group."""
    rewards = rewards.reshape(-1, k)
    sum_all = rewards.sum(dim=1, keepdim=True)
    baseline = (sum_all - rewards) / (k - 1)
    A = rewards - baseline
    return A.reshape(-1)


def rloo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 0.2,
) -> torch.Tensor:
    """与 PPO 同 clip，advantage 用 RLOO."""
    ratio = (log_probs_new - log_probs_old).exp()
    A = advantages.unsqueeze(1)
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - eps, 1 + eps) * A
    loss_per_token = -torch.min(surr1, surr2)
    return (loss_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)


def compare_rloo_grpo():
    print("RLOO vs GRPO advantage 对比")
    rewards = torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0])
    k = 8
    # RLOO
    A_rloo = rloo_advantage(rewards, k)
    # GRPO (z-score)
    rewards_g = rewards.reshape(-1, k)
    mean = rewards_g.mean(dim=1, keepdim=True)
    std = rewards_g.std(dim=1, keepdim=True) + 1e-8
    A_grpo = ((rewards_g - mean) / std).reshape(-1)
    print(f"  rewards : {rewards.tolist()}")
    print(f"  RLOO    : {[round(a, 3) for a in A_rloo.tolist()]}")
    print(f"  GRPO    : {[round(a, 3) for a in A_grpo.tolist()]}")
    print(f"  RLOO mean = {A_rloo.mean():.4f}  std = {A_rloo.std():.4f}")
    print(f"  GRPO mean = {A_grpo.mean():.4f}  std = {A_grpo.std():.4f}")


if __name__ == "__main__":
    print("RLOO minimal\n" + "=" * 50)
    compare_rloo_grpo()
