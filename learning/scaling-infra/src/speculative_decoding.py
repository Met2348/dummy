"""Speculative Decoding - 教学模拟.

不真跑模型, 用 token 概率分布演示 accept/reject 逻辑.
"""
from __future__ import annotations

import random
import torch


def speculative_decode(draft_probs: list, target_probs: list,
                       k: int = 4) -> tuple:
    """简化 speculative decoding.

    Args:
        draft_probs: K 步 draft 生成的 token 概率 [(token, p_draft)]
        target_probs: target 一次 forward 在每位置的 distribution
    Returns:
        accepted_tokens, n_accepted, resampled_token
    """
    accepted = []
    for i in range(min(k, len(draft_probs))):
        tok, p_d = draft_probs[i]
        p_t = target_probs[i].get(tok, 0.0)
        r = random.random()
        if r < min(1.0, p_t / max(p_d, 1e-9)):
            accepted.append(tok)
        else:
            resampled = max(target_probs[i].items(), key=lambda x: x[1])[0]
            return accepted, len(accepted), resampled
    return accepted, len(accepted), None


def speedup(accept_rate: float, K: int = 4,
            target_latency: float = 1.0,
            draft_latency: float = 0.2) -> float:
    """理论 speedup."""
    expected_accept = accept_rate * K
    total_time = K * draft_latency + target_latency
    serial_time = (expected_accept + 1) * target_latency
    return serial_time / total_time


def medusa_heads_count(target_speed: float = 50,
                       desired_speedup: float = 2.5) -> int:
    """估算 Medusa 需要多少 head."""
    return max(1, int(desired_speedup * 2))


if __name__ == "__main__":
    print("=== Speedup curve ===")
    for p in [0.5, 0.7, 0.8, 0.9, 0.95]:
        s = speedup(accept_rate=p, K=4)
        print(f"  accept_rate={p}  K=4  → speedup={s:.2f}×")

    print("\n=== Speculative decode mock ===")
    draft = [(1234, 0.8), (5678, 0.9), (9012, 0.7), (3456, 0.6)]
    target = [
        {1234: 0.85, 9999: 0.1},
        {5678: 0.92, 9999: 0.05},
        {9012: 0.4, 7777: 0.5},
        {3456: 0.7, 1111: 0.2},
    ]
    random.seed(42)
    acc, n, rsp = speculative_decode(draft, target, k=4)
    print(f"  accepted: {acc}  (n={n})  resampled={rsp}")
