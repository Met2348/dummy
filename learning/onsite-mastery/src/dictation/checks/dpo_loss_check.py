"""dpo_loss_spec 的纯断言检验：不 import solutions/。
用 math.log/math.exp 现场手推期望值做独立对拍，不复用被测实现的任何代码。
"""
from __future__ import annotations

import math


def _get(target, pc, pr, rc, rr, beta):
    return float(target(pc, pr, rc, rr, beta))


def check(target) -> None:
    # ---- 1) policy 和 ref 完全相同 -> 隐式奖励差恒为 0 -> loss == -log(0.5) == ln(2) ----
    # 特意让 chosen/rejected 的绝对 log-prob 差别很大（-2 vs -10），只要 policy==ref，
    # 结果也必须和 chosen==rejected 时一样是 ln(2)——专门揪"忘了减 ref 项"的 bug。
    for beta in (0.1, 0.5, 1.0):
        got = _get(target, -2.0, -10.0, -2.0, -10.0, beta)
        assert math.isclose(got, math.log(2.0), rel_tol=1e-6, abs_tol=1e-8), (
            f"policy==ref 时 loss 必须恒等于 ln(2)={math.log(2.0):.6f}（与 chosen/rejected 绝对值无关，"
            f"与 beta 无关），beta={beta} 时实得 {got}——是不是漏减了 ref 项？"
        )

    # ---- 2) 精确数值对拍：ref 都为 0，chosen=1,rejected=0,beta=0.5 ----
    # logits = 0.5 * ((1-0)-(0-0)) = 0.5
    # loss = log(1+exp(-0.5))  (独立手推，不调用被测公式)
    logits = 0.5
    want = math.log(1.0 + math.exp(-logits))
    got = _get(target, 1.0, 0.0, 0.0, 0.0, 0.5)
    assert math.isclose(got, want, rel_tol=1e-6), f"精确数值对拍失败: got={got}, want={want}"

    # ---- 3) 单调性(chosen 方向): 固定 rejected/ref, chosen 隐式奖励越大 loss 应越小 ----
    beta = 0.5
    losses_chosen = [_get(target, c, 0.0, 0.0, 0.0, beta) for c in (0.0, 1.0, 2.0, 4.0)]
    assert losses_chosen == sorted(losses_chosen, reverse=True), (
        f"chosen 隐式奖励越大 loss 应严格递减，实得 {losses_chosen}（方向反了？chosen/rejected 顺序写反了？）"
    )
    assert losses_chosen[0] > losses_chosen[-1] + 1e-6, "chosen 变化应该让 loss 有实质性差异，不能没反应"

    # ---- 4) 单调性(rejected 方向): 固定 chosen/ref, rejected 隐式奖励越大 loss 应越大 ----
    losses_rejected = [_get(target, 0.0, r, 0.0, 0.0, beta) for r in (0.0, 1.0, 2.0, 4.0)]
    assert losses_rejected == sorted(losses_rejected), (
        f"rejected 隐式奖励越大(margin 越小) loss 应严格递增，实得 {losses_rejected}"
    )
    assert losses_rejected[-1] > losses_rejected[0] + 1e-6, "rejected 变化应该让 loss 有实质性差异，不能没反应"
