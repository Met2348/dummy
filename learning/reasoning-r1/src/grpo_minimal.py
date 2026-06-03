"""手写 GRPO — Group Relative Policy Optimization (DeepSeek 2024.02).

教学目标：
    1. 从 PPO 去 critic + group baseline 推导出 GRPO
    2. 实现 group z-score advantage
    3. 实现 unbiased KL estimator (loss 外)

⚠️  本脚本在 WSL2 跑（vllm rollout 加速；不用也能跑只是慢）。

运行：
    python learning/reasoning-r1/src/grpo_minimal.py --total-iters 10
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_group_advantage(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Group z-score advantage.

    rewards: (B*k,) — B 个 prompt × k 个 response
    return: (B*k,) — 每条 response 一个 z-score 的 advantage
    """
    rewards = rewards.reshape(-1, k)                # (B, k)
    mean = rewards.mean(dim=1, keepdim=True)         # (B, 1)
    std = rewards.std(dim=1, keepdim=True) + 1e-8    # (B, 1)
    A = (rewards - mean) / std                       # (B, k)
    return A.reshape(-1)                              # (B*k,)


def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 0.2,
    beta: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """GRPO loss = L_clip + β · KL (unbiased).

    args:
        log_probs_new/old/ref: (B*k, T-1)
        advantages: (B*k,) — broadcast to all tokens
        response_mask: (B*k, T-1) — 1 表示 response 位
    return:
        loss, stats dict
    """
    # 1. PPO clip surrogate
    ratio = (log_probs_new - log_probs_old).exp()
    A_broadcast = advantages.unsqueeze(1)            # (B*k, 1)
    surr1 = ratio * A_broadcast
    surr2 = ratio.clamp(1 - eps, 1 + eps) * A_broadcast
    L_clip_per_token = -torch.min(surr1, surr2)      # (B*k, T-1)

    # 2. unbiased KL estimator (Schulman) — in loss
    log_r = log_probs_ref - log_probs_new
    kl_per_token = log_r.exp() - log_r - 1            # ≥ 0，0 时退化

    # 3. masked mean over response tokens
    mask_sum = response_mask.sum().clamp(min=1)
    L_clip = (L_clip_per_token * response_mask).sum() / mask_sum
    L_kl = (kl_per_token * response_mask).sum() / mask_sum

    loss = L_clip + beta * L_kl

    return loss, {
        "L_clip": L_clip.item(),
        "L_kl": L_kl.item(),
        "mean_ratio": (ratio * response_mask).sum().item() / mask_sum.item(),
    }


def get_log_probs(model: nn.Module, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_pi = F.log_softmax(logits, dim=-1)
    return log_pi.gather(2, targets.unsqueeze(-1)).squeeze(-1)


def main():
    """演示 GRPO 核心组件可调用（不跑完整训练，只 smoke test）。"""
    print("GRPO minimal — component smoke test")

    # 模拟数据
    B, k, T = 2, 4, 16
    rewards = torch.rand(B * k)
    print(f"rewards: {rewards.tolist()}")

    A = compute_group_advantage(rewards, k)
    print(f"group advantage: {A.tolist()}")
    print(f"  group 1 mean = 0? {A[:k].mean():.4f}")
    print(f"  group 2 mean = 0? {A[k:].mean():.4f}")

    log_probs_new = torch.randn(B * k, T - 1)
    log_probs_old = log_probs_new + 0.01 * torch.randn_like(log_probs_new)
    log_probs_ref = log_probs_old.clone()
    response_mask = torch.ones(B * k, T - 1)

    loss, stats = grpo_loss(
        log_probs_new, log_probs_old, log_probs_ref,
        advantages=A, response_mask=response_mask,
        eps=0.2, beta=0.01,
    )
    print(f"loss = {loss.item():.4f}")
    print(f"stats = {stats}")


if __name__ == "__main__":
    main()
