"""Reward Hacking demo — RM 过拟合到表面特征.

设计一个"长度奖励"的退化 RM，让 actor 学会"生成更长 → reward 上升"，
即使内容质量未变。模拟 Gao 2022 的 over-optimization 现象。
"""
from __future__ import annotations

import torch


def length_rm(response_lens: torch.Tensor, target: float = 50.0) -> torch.Tensor:
    """退化 RM：长度越接近 target，奖励越高（无关质量）."""
    return -((response_lens - target) ** 2) / (target ** 2)


def hackable_reward(quality_scores: torch.Tensor, response_lens: torch.Tensor,
                    alpha: float = 0.3) -> torch.Tensor:
    """混合 RM：(1-α) 质量 + α 长度. α 太大 → hack."""
    q_norm = (quality_scores - quality_scores.mean()) / (quality_scores.std() + 1e-8)
    l_norm = length_rm(response_lens)
    return (1 - alpha) * q_norm + alpha * l_norm


def detect_hacking(rewards_traj: list[float], lens_traj: list[float], window: int = 10) -> dict:
    """检测 hacking 信号：reward 上升 + length drift."""
    if len(rewards_traj) < window * 2:
        return {"detected": False, "reason": "too short"}
    r_early = sum(rewards_traj[:window]) / window
    r_late = sum(rewards_traj[-window:]) / window
    l_early = sum(lens_traj[:window]) / window
    l_late = sum(lens_traj[-window:]) / window
    r_up = r_late > r_early + 0.1
    l_drift = abs(l_late - l_early) > 10
    return {
        "detected": r_up and l_drift,
        "reward_change": r_late - r_early,
        "length_change": l_late - l_early,
        "diagnosis": "reward 涨但 length 也漂移 — 可能 hacking" if (r_up and l_drift) else "OK",
    }


if __name__ == "__main__":
    print("Reward Hacking demo\n" + "=" * 50)
    # 模拟 100 step 训练
    torch.manual_seed(0)
    n_steps = 100
    # 质量基本不变（围绕 0.5 抖动）；actor 学到"把长度推向 RM 偏好的 target=50 → 提分"。
    # 注意：质量归一化必须在整条轨迹上做（单元素 tensor 的 std 为 NaN，会让 reward 全变 NaN）。
    quality = torch.randn(n_steps) * 0.1 + 0.5
    lengths = 20.0 + torch.arange(n_steps).float() * 0.3   # 20 → ~50，漂向退化 RM 的 target
    rewards_t = hackable_reward(quality, lengths, alpha=0.5)
    rewards = rewards_t.tolist()
    lens = lengths.tolist()
    for step in range(0, n_steps, 20):
        print(f"step {step}: quality≈{quality[step].item():.2f}, "
              f"len={lens[step]:.0f}, r={rewards[step]:.3f}")
    diag = detect_hacking(rewards, lens)
    print(f"\n诊断: {diag}")
    print("\n修复建议:")
    print("  1. 降 α (减少长度权重)")
    print("  2. 加 KL ref penalty (防止 actor 漂出 SFT distribution)")
    print("  3. 在 RM 训练数据中加入'同长度对比'削弱长度信号")
