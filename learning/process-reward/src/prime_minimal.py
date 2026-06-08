"""PRIME (Tsinghua 2025.02) — 隐式 PRM，从 outcome 自动学步级 reward.

idea: implicit reward r_t = log π_actor(y_t|y_<t) - log π_ref(y_t|y_<t).
不需要 PRM 训练数据，actor 与 ref 之差自然形成 step-level 信号.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def implicit_prm_per_token(
    log_p_actor: torch.Tensor,
    log_p_ref: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """r_t = β · (log π_actor - log π_ref).

    与 KL penalty 反号 — 这里是 reward 增量.
    """
    return beta * (log_p_actor - log_p_ref)


def aggregate_to_step(per_token_r: torch.Tensor,
                      step_end_positions: list[int]) -> torch.Tensor:
    """累加 per-token reward 到 step end → 步级 reward."""
    cumsum = per_token_r.cumsum(dim=-1)
    step_rewards = []
    prev = 0
    for pos in step_end_positions:
        # 区间 [prev:pos+1] 的 sum
        seg = cumsum[..., pos] - (cumsum[..., prev - 1] if prev > 0 else 0)
        step_rewards.append(seg)
        prev = pos + 1
    return torch.stack(step_rewards, dim=-1)


def prime_advantage(
    outcome_reward: torch.Tensor,        # (B,)
    step_rewards: torch.Tensor,          # (B, n_steps)
    gamma: float = 1.0,
) -> torch.Tensor:
    """PRIME advantage = outcome reward 加权 + step 信号自学."""
    discounted_outcome = outcome_reward.unsqueeze(1) * (gamma ** torch.arange(
        step_rewards.size(1) - 1, -1, -1, device=step_rewards.device).float())
    return step_rewards + discounted_outcome


if __name__ == "__main__":
    print("PRIME minimal - implicit PRM 从 outcome 自动学\n" + "=" * 50)
    torch.manual_seed(0)
    B, T = 2, 12
    log_p_actor = torch.randn(B, T) - 1
    log_p_ref = torch.randn(B, T) - 1
    per_token_r = implicit_prm_per_token(log_p_actor, log_p_ref, beta=0.05)
    print(f"per-token reward shape: {per_token_r.shape}")

    step_end = [3, 7, 11]
    step_r = aggregate_to_step(per_token_r, step_end)
    print(f"step rewards shape: {step_r.shape}")

    outcome = torch.tensor([1.0, -0.5])
    A = prime_advantage(outcome, step_r)
    print(f"\nPRIME advantage:\n{A}")
    print("\nidea: 不训 PRM，actor/ref 之差自然形成 step 级信号")
