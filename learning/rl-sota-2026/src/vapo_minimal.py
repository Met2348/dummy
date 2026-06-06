"""VAPO (ByteDance 2025.04) — Length-Adaptive GAE.

GRPO/PPO 的 GAE 用固定 λ. 短/长 response 信息融合不同:
    短: λ 小 (用更多 outcome)
    长: λ 大 (强 bootstrap)

VAPO 让 λ 随 |y| 动态调整:
    λ(y) = clamp(λ_min + α·log|y|, λ_min, λ_max)
"""
from __future__ import annotations

import torch


def adaptive_lambda(response_lens: torch.Tensor,
                    lam_min: float = 0.5,
                    lam_max: float = 0.99,
                    alpha: float = 0.1) -> torch.Tensor:
    """λ 随 log|y| 增长."""
    log_lens = response_lens.double().clamp(min=1).log()
    return (lam_min + alpha * log_lens).clamp(lam_min, lam_max)


def length_adaptive_gae(
    rewards: torch.Tensor,     # (B, T)
    values: torch.Tensor,       # (B, T)
    dones: torch.Tensor,        # (B, T)
    response_lens: torch.Tensor, # (B,)
    gamma: float = 1.0,
) -> torch.Tensor:
    """对每条 response 用 length-adaptive λ."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    lambdas = adaptive_lambda(response_lens)
    for b in range(B):
        lam_b = lambdas[b].item()
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[b, t].item()
            delta = rewards[b, t] + gamma * next_v * mask - values[b, t]
            gae = delta.item() + gamma * lam_b * mask * gae
            adv[b, t] = gae
            next_v = values[b, t].item()
    return adv


def value_pretraining_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """VAPO 推荐 value head 先单独训 1k step on outcome."""
    return ((values - returns) ** 2).mean()


if __name__ == "__main__":
    print("VAPO minimal — Length-Adaptive GAE\n" + "=" * 50)
    torch.manual_seed(0)
    B, T = 3, 50
    rewards = torch.zeros(B, T)
    rewards[:, -1] = torch.tensor([1.0, 0.5, -0.5])
    values = torch.randn(B, T) * 0.1
    dones = torch.zeros(B, T); dones[:, -1] = 1
    lens = torch.tensor([10.0, 30.0, 200.0])
    lams = adaptive_lambda(lens)
    print(f"response lens: {lens.tolist()}")
    print(f"adaptive λ   : {[round(l, 4) for l in lams.tolist()]}")
    adv = length_adaptive_gae(rewards, values, dones, lens)
    print(f"advantage[:, -3:]:\n{adv[:, -3:]}")
    print("\n关键: 长 response 用 λ→0.99 (更 bootstrap)，短用 λ→0.5 (outcome-heavy)")
