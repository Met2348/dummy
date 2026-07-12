"""grpo_advantage_spec 的纯断言检验：不 import solutions/。"""
from __future__ import annotations

import math


def check(target) -> None:
    # ---- 1) 一般情况: 输出均值应约等于 0（标准化的定义性质） ----
    rewards = [1.0, 2.0, 3.0, 4.0]
    out = target(rewards)
    assert len(out) == len(rewards), f"输出长度应和输入一致, got {len(out)}"
    assert all(math.isfinite(v) for v in out), f"输出不应有 NaN/Inf: {out}"
    mean_out = sum(out) / len(out)
    assert abs(mean_out) < 1e-6, f"标准化后均值应约为 0, 实得 {mean_out}"

    # 保序性：reward 越大, advantage 也应该越大（同一个单调标准化变换不改变相对顺序）
    order = sorted(range(len(rewards)), key=lambda i: rewards[i])
    out_in_reward_order = [out[i] for i in order]
    assert out_in_reward_order == sorted(out_in_reward_order), (
        f"advantage 的相对顺序必须和 reward 的顺序一致, rewards={rewards}, out={out}"
    )
    # 最大 reward 的 advantage 应为正，最小的应为负（reward 有明显差异时）
    assert out[0] < 0 < out[-1], f"最小 reward 的 advantage 应<0，最大的应>0，实得 {out}"

    # ---- 2) 组内 reward 全相同 -> std=0，eps 防护生效，advantage 应精确全为 0（不是 NaN） ----
    same = [5.0, 5.0, 5.0, 5.0]
    out_same = target(same)
    assert len(out_same) == 4
    for v in out_same:
        assert math.isfinite(v), f"reward 全相同时不应出现 NaN/Inf（eps 防护没生效？）: {out_same}"
        assert v == 0.0, f"reward 全相同时 advantage 应精确为 0，实得 {out_same}"

    # ---- 3) 单元素组的边界情况：std 天然是 0，同样要靠 eps 防护不崩 ----
    single = [42.0]
    out_single = target(single)
    assert len(out_single) == 1
    assert math.isfinite(out_single[0])
    assert out_single[0] == 0.0, f"单元素组 advantage 应为 0，实得 {out_single}"

    # ---- 4) 负数 reward 也要正确处理（不依赖 reward 全正的假设） ----
    neg = [-3.0, -1.0, 1.0, 3.0]
    out_neg = target(neg)
    mean_neg = sum(out_neg) / len(out_neg)
    assert abs(mean_neg) < 1e-6, f"负数 reward 标准化后均值也应约为 0, 实得 {mean_neg}"
    assert out_neg[0] < 0 < out_neg[-1]
