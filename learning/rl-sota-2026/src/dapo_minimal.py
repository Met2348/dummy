"""DAPO 4 件套独立实现 + 组合 — 基于 GRPO 增量.

继承 reasoning-r1/grpo_minimal.py。

4 件套:
    1. Clip-Higher: ε_high > ε_low (不对称 clip)
    2. Dynamic Sampling: rollout 直到有对有错
    3. Token-level PG Loss: mean over (B*k*T) 而非 (B*k)
    4. Overlong Reward Shaping: sigmoid soft penalty
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ===== Trick 1: Clip-Higher =====

def asymmetric_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> torch.Tensor:
    """Clip-Higher: ratio ∈ [1-eps_low, 1+eps_high]."""
    ratio = (log_probs_new - log_probs_old).exp()
    A = advantages.unsqueeze(1)
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - eps_low, 1 + eps_high) * A
    return -torch.min(surr1, surr2)        # per-token (B*k, T-1)


# ===== Trick 2: Dynamic Sampling =====

def is_group_useful(rewards: torch.Tensor) -> bool:
    """Group 是否有对有错（advantage 非零）."""
    return 0 < rewards.sum() < len(rewards)


def dynamic_sampling_rollout(rollout_fn, prompt, k_min: int = 8, k_max: int = 32):
    """对单 prompt 持续 rollout 直到 group 有对有错或达到 k_max."""
    rollouts = []
    rewards_so_far = []
    while True:
        new_resp, new_rewards = rollout_fn(prompt, n=k_min)
        rollouts.extend(new_resp)
        rewards_so_far.extend(new_rewards.tolist())
        if is_group_useful(torch.tensor(rewards_so_far)):
            break
        if len(rollouts) >= k_max:
            break
    return rollouts, torch.tensor(rewards_so_far)


# ===== Trick 3: Token-level PG Loss =====

def token_level_loss(per_token_loss: torch.Tensor,
                     response_mask: torch.Tensor) -> torch.Tensor:
    """每 token 同等权重 (不再 / response_len)。"""
    return (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1)


def response_level_loss(per_token_loss: torch.Tensor,
                        response_mask: torch.Tensor) -> torch.Tensor:
    """每 response 同等权重（标准 GRPO）。"""
    per_resp_loss = (per_token_loss * response_mask).sum(dim=1) / \
                    response_mask.sum(dim=1).clamp(min=1)
    return per_resp_loss.mean()


# ===== Trick 4: Overlong Reward Shaping =====

def overlong_shaping(
    rewards: torch.Tensor,
    response_lens: torch.Tensor,
    target_len: int = 4096,
    alpha: float = 200.0,
) -> torch.Tensor:
    """长度超过 target_len 时 reward 用 sigmoid soft penalty 衰减.

    args:
        rewards: (B*k,)
        response_lens: (B*k,)
    """
    over = (response_lens >= target_len).float()
    penalty_factor = torch.sigmoid((target_len - response_lens.float()) / alpha)
    # over 段用 penalty，未 over 段保留原 reward
    return rewards * (1 - over) + rewards * penalty_factor * over


# ===== 测试 / 演示 =====

def _self_test():
    print("DAPO 4 件套 smoke test\n")

    # 1. Clip-Higher
    log_p_new = torch.randn(4, 5)
    log_p_old = log_p_new + 0.05 * torch.randn_like(log_p_new)
    A = torch.tensor([1.0, -0.5, 0.3, -0.8])
    mask = torch.ones(4, 5)
    L_high = asymmetric_clip_loss(log_p_new, log_p_old, A, mask,
                                  eps_low=0.2, eps_high=0.28)
    L_sym = asymmetric_clip_loss(log_p_new, log_p_old, A, mask,
                                  eps_low=0.2, eps_high=0.2)
    print(f"Clip-Higher L shape: {L_high.shape}")
    print(f"  asymmetric mean: {L_high.mean():.4f}")
    print(f"  symmetric  mean: {L_sym.mean():.4f}")

    # 2. Dynamic sampling
    print(f"\nGroup useful?")
    print(f"  [1,1,1,1] → {is_group_useful(torch.tensor([1.0, 1, 1, 1]))}")  # False
    print(f"  [0,0,0,0] → {is_group_useful(torch.tensor([0.0, 0, 0, 0]))}")  # False
    print(f"  [1,0,1,0] → {is_group_useful(torch.tensor([1.0, 0, 1, 0]))}")  # True

    # 3. Token-level vs response-level
    per_token = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],    # response len 3
        [1.0, 1.0, 1.0, 1.0, 1.0],    # response len 5
    ])
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)
    L_t = token_level_loss(per_token, mask)
    L_r = response_level_loss(per_token, mask)
    print(f"\nToken-level loss = {L_t:.4f}  (8/8=1.0)")
    print(f"Response-level loss = {L_r:.4f}  ((1.0+1.0)/2=1.0)")
    print("  长度对称时两者相同；不对称时 token-level 给长 response 更多权重")

    # 4. Overlong Shaping
    rewards = torch.tensor([1.0, 1.0, 1.0])
    lens = torch.tensor([2000.0, 4096.0, 6000.0])
    shaped = overlong_shaping(rewards, lens, target_len=4096, alpha=200)
    print(f"\nOverlong shaping rewards: {shaped.tolist()}")
    print("  short 2000 → 保留 1.0")
    print("  exact 4096 → sigmoid(0) = 0.5")
    print("  long 6000 → 接近 0")


if __name__ == "__main__":
    _self_test()
