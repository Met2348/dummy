"""ppo_clip_spec 的纯断言检验：不 import solutions/，期望值全部手算好写成字面量。"""
from __future__ import annotations

import math


def _get(target, ratio, advantage, clip_eps):
    out = target(ratio, advantage, clip_eps)
    return float(out)


def check(target) -> None:
    eps = 0.2

    # ---- 1) 信任域内 [1-eps, 1+eps]：必须精确退化为未裁剪值 ratio*A，无论 A 正负 ----
    for ratio in (0.8, 0.9, 1.0, 1.1, 1.2):
        for adv in (2.0, -3.0):
            got = _get(target, ratio, adv, eps)
            want = ratio * adv
            assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9), (
                f"信任域内应退化为未裁剪值 ratio*A: ratio={ratio}, adv={adv}, got={got}, want={want}"
            )

    # ---- 2) ratio > 1+eps 且 A > 0 -> clip 生效（min 选中更小的裁剪值 2.4，而不是未裁剪的 3.0）----
    got = _get(target, 1.5, 2.0, eps)
    assert math.isclose(got, 2.4, rel_tol=1e-9), (
        f"ratio=1.5(>1.2), A=2.0>0 时应裁剪生效 -> 2.4，实得 {got}（是不是 min/max 用反了？）"
    )

    # ---- 3) ratio > 1+eps 且 A < 0 -> clip 不生效（min 选中未裁剪的 -3.0，而不是 -2.4）----
    got = _get(target, 1.5, -2.0, eps)
    assert math.isclose(got, -3.0, rel_tol=1e-9), (
        f"ratio=1.5(>1.2), A=-2.0<0 时 clip 不该生效，应保留未裁剪值 -3.0，实得 {got}"
    )

    # ---- 4) ratio < 1-eps 且 A < 0 -> clip 生效（min 选中更小的裁剪值 -1.6，而不是未裁剪的 -1.0）----
    got = _get(target, 0.5, -2.0, eps)
    assert math.isclose(got, -1.6, rel_tol=1e-9), (
        f"ratio=0.5(<0.8), A=-2.0<0 时应裁剪生效 -> -1.6，实得 {got}"
    )

    # ---- 5) ratio < 1-eps 且 A > 0 -> clip 不生效（min 选中未裁剪的 1.0，而不是 1.6）----
    got = _get(target, 0.5, 2.0, eps)
    assert math.isclose(got, 1.0, rel_tol=1e-9), (
        f"ratio=0.5(<0.8), A=2.0>0 时 clip 不该生效，应保留未裁剪值 1.0，实得 {got}"
    )

    # ---- 6) 边界连续性：ratio 恰好等于 1+eps / 1-eps 时，clip 后等于 ratio 本身，两支应完全相等 ----
    for ratio in (0.8, 1.2):
        for adv in (5.0, -5.0):
            got = _get(target, ratio, adv, eps)
            want = ratio * adv
            assert math.isclose(got, want, rel_tol=1e-9), (
                f"边界 ratio={ratio} 处应连续（未裁剪=裁剪），got={got}, want={want}"
            )
