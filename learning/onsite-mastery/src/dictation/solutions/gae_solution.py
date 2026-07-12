"""gae_spec 的参考实现。纯 python，从后往前递推。"""
from __future__ import annotations


def gae_advantage(rewards: list[float], values: list[float], gamma: float, lam: float) -> list[float]:
    t_len = len(rewards)
    assert len(values) == t_len + 1, "values 长度必须是 rewards 长度 + 1（多一个 bootstrap 值）"

    advantages = [0.0] * t_len
    running = 0.0
    for t in range(t_len - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages
