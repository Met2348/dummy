"""grpo_advantage_spec 的参考实现。纯 python，population std（ddof=0）+ eps 防护。"""
from __future__ import annotations


def grpo_group_advantage(rewards: list[float], eps: float = 1e-4) -> list[float]:
    n = len(rewards)
    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n  # population std: 除以 n 不是 n-1
    std_r = variance ** 0.5
    denom = std_r + eps
    return [(r - mean_r) / denom for r in rewards]
