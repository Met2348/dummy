"""Idea 10 pilot:想象到底什么时候真的有用?

在 SlipperyGridWorld 上,对比"信任已经烘焙好的价值函数"(no-imagination) vs
"决策时用同一个不完美模型做 Monte Carlo 想象搜索"(fixed-budget imagination),
用真实环境算出的 Q* 当裁判,统计:
  - 决策改变率:想象改变了多大比例的决策
  - 命中率/帮倒忙率:改变的决策里,有多少真的选到了更好的动作、有多少选到了更差的
  - 和模型置信度(visit_count 代理)的关联:想象是不是在模型没把握的地方更有用/更危险

这是原型规模的诊断性 pilot,不是论文最终实验(见 ../PROTOCOL.md 的范围声明)。
"""
from __future__ import annotations

import random
import statistics

from gridworld_env import ACTIONS, N_STATES, is_terminal, value_iteration, true_transition_dist
from world_model import LearnedWorldModel, collect_rollout_data
from imagination_planner import imagine_action, no_imagination_action

N_TRAIN_TRANSITIONS = 400
SEEDS = [0, 1, 2, 3, 4]
H_SWEEP = [1, 2, 3, 5, 8]
K_FIXED_FOR_H_SWEEP = 5
K_SWEEP = [1, 3, 5, 10]
H_FIXED_FOR_K_SWEEP = 3
UNCERTAINTY_K, UNCERTAINTY_H = 10, 5  # 用于不确定性关联分析的固定慷慨预算


def eval_states() -> list[int]:
    return [s for s in range(N_STATES) if not is_terminal(s)]


def run_one_setting(model, Vhat, Qstar, K, H, rng, states) -> dict:
    changed = hit = hurt = 0
    for s in states:
        a0 = no_imagination_action(model, Vhat, s)
        a1 = imagine_action(model, Vhat, s, K, H, rng)
        if a1 != a0:
            changed += 1
            if Qstar[(s, a1)] > Qstar[(s, a0)] + 1e-9:
                hit += 1
            elif Qstar[(s, a1)] < Qstar[(s, a0)] - 1e-9:
                hurt += 1
    n = len(states)
    return {
        "decision_change_rate": changed / n,
        "hit_rate_among_changed": (hit / changed) if changed else None,
        "hurt_rate_among_changed": (hurt / changed) if changed else None,
        "n_changed": changed,
    }


def uncertainty_bucket_analysis(model, Vhat, Qstar, rng, states) -> dict:
    """按 avg visit_count 中位数分高/低两组,比较想象"命中"比例是否和模型置信度相关。"""
    records = []
    for s in states:
        avg_visits = sum(model.visit_count(s, a) for a in ACTIONS) / len(ACTIONS)
        a0 = no_imagination_action(model, Vhat, s)
        a1 = imagine_action(model, Vhat, s, UNCERTAINTY_K, UNCERTAINTY_H, rng)
        outcome = "no_change"
        if a1 != a0:
            if Qstar[(s, a1)] > Qstar[(s, a0)] + 1e-9:
                outcome = "hit"
            elif Qstar[(s, a1)] < Qstar[(s, a0)] - 1e-9:
                outcome = "hurt"
            else:
                outcome = "neutral_change"
        records.append((avg_visits, outcome))

    visit_counts = sorted(v for v, _ in records)
    median = visit_counts[len(visit_counts) // 2]
    low_conf = [o for v, o in records if v <= median]
    high_conf = [o for v, o in records if v > median]

    def rate(group, target):
        return group.count(target) / len(group) if group else None

    return {
        "median_visit_count": median,
        "low_confidence_n": len(low_conf),
        "high_confidence_n": len(high_conf),
        "low_confidence_hit_rate": rate(low_conf, "hit"),
        "low_confidence_hurt_rate": rate(low_conf, "hurt"),
        "high_confidence_hit_rate": rate(high_conf, "hit"),
        "high_confidence_hurt_rate": rate(high_conf, "hurt"),
    }


def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}" if mean is not None else "n/a"


def main() -> None:
    _, Qstar = value_iteration(true_transition_dist)
    states = eval_states()
    assert len(states) == 32

    results_H = {H: [] for H in H_SWEEP}
    results_K = {K: [] for K in K_SWEEP}
    uncertainty_results = []

    for seed in SEEDS:
        rng = random.Random(seed)
        data = collect_rollout_data(N_TRAIN_TRANSITIONS, rng)
        model = LearnedWorldModel(data)
        Vhat, _ = value_iteration(model.transition_dist)

        for H in H_SWEEP:
            results_H[H].append(run_one_setting(model, Vhat, Qstar, K_FIXED_FOR_H_SWEEP, H, rng, states))
        for K in K_SWEEP:
            results_K[K].append(run_one_setting(model, Vhat, Qstar, K, H_FIXED_FOR_K_SWEEP, rng, states))
        uncertainty_results.append(uncertainty_bucket_analysis(model, Vhat, Qstar, rng, states))

    print(f"=== 扫 H(固定 K={K_FIXED_FOR_H_SWEEP})，{len(SEEDS)} 个种子取均值±标准差 ===")
    print(f"{'H':>3} | {'决策改变率':>14} | {'命中率(改变里)':>16} | {'帮倒忙率(改变里)':>18}")
    for H in H_SWEEP:
        rs = results_H[H]
        cr_m, cr_s = statistics.mean(r["decision_change_rate"] for r in rs), statistics.pstdev(
            r["decision_change_rate"] for r in rs
        )
        hits = [r["hit_rate_among_changed"] for r in rs if r["hit_rate_among_changed"] is not None]
        hurts = [r["hurt_rate_among_changed"] for r in rs if r["hurt_rate_among_changed"] is not None]
        hr = fmt(statistics.mean(hits), statistics.pstdev(hits)) if hits else "n/a"
        ur = fmt(statistics.mean(hurts), statistics.pstdev(hurts)) if hurts else "n/a"
        print(f"{H:>3} | {fmt(cr_m, cr_s):>14} | {hr:>16} | {ur:>18}")

    print(f"\n=== 扫 K(固定 H={H_FIXED_FOR_K_SWEEP})，{len(SEEDS)} 个种子取均值±标准差 ===")
    print(f"{'K':>3} | {'决策改变率':>14} | {'命中率(改变里)':>16} | {'帮倒忙率(改变里)':>18}")
    for K in K_SWEEP:
        rs = results_K[K]
        cr_m, cr_s = statistics.mean(r["decision_change_rate"] for r in rs), statistics.pstdev(
            r["decision_change_rate"] for r in rs
        )
        hits = [r["hit_rate_among_changed"] for r in rs if r["hit_rate_among_changed"] is not None]
        hurts = [r["hurt_rate_among_changed"] for r in rs if r["hurt_rate_among_changed"] is not None]
        hr = fmt(statistics.mean(hits), statistics.pstdev(hits)) if hits else "n/a"
        ur = fmt(statistics.mean(hurts), statistics.pstdev(hurts)) if hurts else "n/a"
        print(f"{K:>3} | {fmt(cr_m, cr_s):>14} | {hr:>16} | {ur:>18}")

    print(f"\n=== 不确定性(visit_count 代理)分组分析,固定 K={UNCERTAINTY_K},H={UNCERTAINTY_H} ===")
    low_hit = [r["low_confidence_hit_rate"] for r in uncertainty_results if r["low_confidence_hit_rate"] is not None]
    low_hurt = [r["low_confidence_hurt_rate"] for r in uncertainty_results if r["low_confidence_hurt_rate"] is not None]
    high_hit = [r["high_confidence_hit_rate"] for r in uncertainty_results if r["high_confidence_hit_rate"] is not None]
    high_hurt = [
        r["high_confidence_hurt_rate"] for r in uncertainty_results if r["high_confidence_hurt_rate"] is not None
    ]
    print(f"低置信度(visit_count 低于中位数)状态组: 命中率={fmt(statistics.mean(low_hit), statistics.pstdev(low_hit)) if low_hit else 'n/a'}"
          f"  帮倒忙率={fmt(statistics.mean(low_hurt), statistics.pstdev(low_hurt)) if low_hurt else 'n/a'}")
    print(f"高置信度(visit_count 高于中位数)状态组: 命中率={fmt(statistics.mean(high_hit), statistics.pstdev(high_hit)) if high_hit else 'n/a'}"
          f"  帮倒忙率={fmt(statistics.mean(high_hurt), statistics.pstdev(high_hurt)) if high_hurt else 'n/a'}")


if __name__ == "__main__":
    main()
