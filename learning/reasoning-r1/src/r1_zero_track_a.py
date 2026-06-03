"""Capstone Track A — GPT-2-M + Countdown-3 教学轨.

目标: 跑通 R1-Zero 完整 pipeline
    base: GPT-2-medium (355M)
    task: Countdown-3 (3 数四则运算凑 target)
    reward: 0.1 * format + 0.9 * accuracy
    algo: GRPO (k=8)
    显存: 5090 24GB OK
    时长: 4-6h
预期: format 5%→95%, accuracy 5%→15-25%, len 50→150
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).parent
sys.path.insert(0, str(REPO_SRC))

from grpo_minimal import compute_group_advantage, grpo_loss   # noqa
from rewards.format_reward import format_reward                # noqa
from rewards.accuracy_reward import countdown_reward           # noqa


# ===== Countdown-3 数据生成 =====

def gen_countdown_problem(rng: random.Random) -> tuple[list[int], int]:
    """生成可解的 Countdown-3 题."""
    a, b, c = rng.sample(range(1, 20), 3)
    op1 = rng.choice("+-*")
    op2 = rng.choice("+-*")
    target = eval(f"{a}{op1}{b}{op2}{c}")
    return [a, b, c], target


def build_prompt(nums: list[int], target: int) -> str:
    n_str = ", ".join(map(str, nums))
    return (
        f"Use the numbers {n_str} with +, -, *, / and parentheses to make {target}.\n"
        f"Show your reasoning in <think>...</think> then the equation in <answer>...</answer>.\n"
    )


def combined_reward(response: str, target: int, nums: list[int],
                    alpha: float = 0.1) -> dict:
    f = format_reward(response)
    # extract answer
    m = re.search(r"<answer>(.+?)</answer>", response, re.DOTALL)
    a = 0.0
    if m:
        a = countdown_reward(m.group(1), nums, target)
    return {"format": f, "accuracy": a, "total": alpha * f + (1 - alpha) * a}


# ===== 训练 loop (mock - 真训需 actor + tokenizer) =====

def mock_train_step(step: int, k: int = 8):
    """模拟一步 GRPO 训练，返回 metrics."""
    # 真实: rollout k 条 + 算 reward + GRPO update
    # 这里只演示 reward 与 advantage 计算
    rng = random.Random(step)
    rewards = []
    for _ in range(k):
        # 模拟 reward 随训练涨
        base = 0.05 + step * 0.001
        r = 1.0 if rng.random() < base else 0.0
        rewards.append(r)
    rewards_t = torch.tensor(rewards)
    A = compute_group_advantage(rewards_t, k)
    return {
        "step": step,
        "mean_reward": rewards_t.mean().item(),
        "adv_std": A.std().item(),
        "any_positive": (rewards_t > 0).any().item(),
    }


def train_track_a(total_steps: int = 200, k: int = 8):
    print(f"Track A — GPT-2-M Countdown-3 GRPO mock\n{'='*50}")
    print(f"  steps={total_steps}, k={k}")
    history = []
    for step in range(total_steps):
        m = mock_train_step(step, k)
        history.append(m)
        if step % 20 == 0:
            print(f"  step {step:4d}: mean_R={m['mean_reward']:.2f} "
                  f"adv_std={m['adv_std']:.3f}")
    early = sum(h["mean_reward"] for h in history[:20]) / 20
    late = sum(h["mean_reward"] for h in history[-20:]) / 20
    print(f"\n  early avg reward = {early:.3f}")
    print(f"  late  avg reward = {late:.3f}")
    print(f"  Δ = +{(late - early)*100:.1f}pp")
    return history


if __name__ == "__main__":
    print("Capstone Track A — 教学轨\n")
    # 1. 数据演示
    rng = random.Random(42)
    nums, target = gen_countdown_problem(rng)
    print(f"Problem: {nums} → target={target}")
    prompt = build_prompt(nums, target)
    print(f"Prompt:\n{prompt[:120]}\n")
    response = f"<think>{nums[0]}+{nums[1]} = {nums[0]+nums[1]}, then *{nums[2]}={target}</think>" \
               f"<answer>({nums[0]}+{nums[1]})*{nums[2]}={target}</answer>"
    r = combined_reward(response, target, nums)
    print(f"Mock reward: {r}\n")
    # 2. mock 训练
    train_track_a(200, 8)
