"""REINFORCE++ (OpenRLHF 2025.01) — 简化 PPO，省 critic + KL penalty 灵活.

变化:
    1. 去 critic (用 group baseline 或 batch baseline)
    2. KL penalty 加在 reward 而非 loss
    3. 保留 PPO clip
    4. 简化 GAE 至单步 advantage
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reinforce_pp_advantage(
    rewards: torch.Tensor,
    kl_per_token: torch.Tensor,
    response_mask: torch.Tensor,
    beta_kl: float = 0.04,
) -> torch.Tensor:
    """A = (R - KL_penalty) - batch_mean.

    R 是 episode-level reward, KL_penalty 是累积 KL.
    """
    cumulative_kl = (kl_per_token * response_mask).sum(dim=1)
    R_corrected = rewards - beta_kl * cumulative_kl
    baseline = R_corrected.mean()
    return R_corrected - baseline


def reinforce_pp_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 0.2,
) -> torch.Tensor:
    ratio = (log_probs_new - log_probs_old).exp()
    A = advantages.unsqueeze(1)
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - eps, 1 + eps) * A
    loss = -torch.min(surr1, surr2)
    return (loss * response_mask).sum() / response_mask.sum().clamp(min=1)


if __name__ == "__main__":
    print("REINFORCE++ minimal — smoke test\n" + "=" * 50)
    torch.manual_seed(0)
    B, T = 4, 10
    rewards = torch.tensor([1.0, 0.5, -0.5, 1.0])
    kl_per_token = torch.randn(B, T).abs() * 0.01
    response_mask = torch.ones(B, T)
    A = reinforce_pp_advantage(rewards, kl_per_token, response_mask)
    print(f"rewards     : {rewards.tolist()}")
    print(f"advantages  : {[round(a, 3) for a in A.tolist()]}")

    log_p_new = torch.randn(B, T) - 1
    log_p_old = log_p_new.clone() + 0.01 * torch.randn_like(log_p_new)
    L = reinforce_pp_loss(log_p_new, log_p_old, A, response_mask)
    print(f"loss        : {L.item():.4f}")
    print("\n核心: 省 critic + KL 在 reward + PPO clip 保留")
