"""Idea 10 pilot——神经网络集成世界模型版本(对照 run_pilot_study.py 的表格模型版本)。

在动手跑之前先把预测写下来(见 RESULTS-neural-ensemble.md 的"预注册预测"一节),这里只放实现。

回答两个问题:
1. sample_mode="consensus":和表格 pilot 完全同源的设计(想象只能看到模型已经烘焙进 Vhat 的
   同一份信息),只换模型类(表格计数 -> 神经网络 ensemble)。如果发现一的 Bellman telescoping
   论证是对的,这里应该复现"H越深命中率不升反降"的同一个模式。
2. sample_mode="member":想象 rollout 时每步随机抽一个集成成员而不是用集成平均——这时候想象能
   看到基线 Vhat(用集成平均烘焙)看不到的东西:集成内部的分歧。这是不是能让想象从净负翻正?
3. 不确定性分组分析换成 ensemble_disagreement(集成分歧),替代表格 pilot 用的 visit_count,
   重新检验"高不确定性状态更容易被想象帮倒忙"这个直觉这次站不站得住。

运行:python -X utf8 run_pilot_study_neural.py
(比表格版本慢——每个种子要训练 5 个小神经网络,预计几十秒量级,仍是可控的 CPU 任务)。
"""
from __future__ import annotations

import random
import statistics
import time

from gridworld_env import ACTIONS, N_STATES, is_terminal, value_iteration, true_transition_dist
from world_model import collect_rollout_data
from imagination_planner import imagine_action, no_imagination_action
from neural_ensemble_model import NeuralEnsembleWorldModel

N_TRAIN_TRANSITIONS = 400
SEEDS = [0, 1, 2, 3, 4]
H_SWEEP = [1, 2, 3, 5, 8]
K_FIXED_FOR_H_SWEEP = 5
K_SWEEP = [1, 3, 5, 10]
H_FIXED_FOR_K_SWEEP = 3
UNCERTAINTY_K, UNCERTAINTY_H = 10, 5


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
    }


def disagreement_bucket_analysis(model, Vhat, Qstar, rng, states) -> dict:
    """用 model(固定 sample_mode="consensus")的 ensemble_disagreement 当不确定性代理分组——
    和表格 pilot 的 uncertainty_bucket_analysis 对应,只是把 visit_count 换成 ensemble_disagreement。
    """
    records = []
    for s in states:
        avg_dis = sum(model.ensemble_disagreement(s, a) for a in ACTIONS) / len(ACTIONS)
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
        records.append((avg_dis, outcome))

    vals = sorted(v for v, _ in records)
    median = vals[len(vals) // 2]
    low_conf = [o for v, o in records if v > median]  # 分歧大 = 集成内部意见不一 = "低置信度"
    high_conf = [o for v, o in records if v <= median]  # 分歧小 = "高置信度"

    def rate(group, target):
        return group.count(target) / len(group) if group else None

    return {
        "low_confidence_hit_rate": rate(low_conf, "hit"),
        "low_confidence_hurt_rate": rate(low_conf, "hurt"),
        "high_confidence_hit_rate": rate(high_conf, "hit"),
        "high_confidence_hurt_rate": rate(high_conf, "hurt"),
    }


def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}" if mean is not None else "n/a"


def summarize(results_by_key, keys):
    for key in keys:
        rs = results_by_key[key]
        cr_m = statistics.mean(r["decision_change_rate"] for r in rs)
        cr_s = statistics.pstdev(r["decision_change_rate"] for r in rs)
        hits = [r["hit_rate_among_changed"] for r in rs if r["hit_rate_among_changed"] is not None]
        hurts = [r["hurt_rate_among_changed"] for r in rs if r["hurt_rate_among_changed"] is not None]
        hr = fmt(statistics.mean(hits), statistics.pstdev(hits)) if hits else "n/a"
        ur = fmt(statistics.mean(hurts), statistics.pstdev(hurts)) if hurts else "n/a"
        print(f"{key:>3} | {fmt(cr_m, cr_s):>14} | {hr:>16} | {ur:>18}")


def main() -> None:
    t0 = time.time()
    _, Qstar = value_iteration(true_transition_dist)
    states = eval_states()
    assert len(states) == 32

    results = {mode: {"H": {H: [] for H in H_SWEEP}, "K": {K: [] for K in K_SWEEP}} for mode in ("consensus", "member")}
    disagreement_results = []

    for seed in SEEDS:
        data = collect_rollout_data(N_TRAIN_TRANSITIONS, random.Random(seed))
        trained = NeuralEnsembleWorldModel(data, seed=seed, sample_mode="consensus")  # 只训练一次
        Vhat, _ = value_iteration(trained.transition_dist)  # Vhat定义和sample_mode无关

        disagreement_results.append(
            disagreement_bucket_analysis(trained, Vhat, Qstar, random.Random(seed * 7 + 1), states)
        )

        for mode in ("consensus", "member"):
            view = trained.with_sample_mode(mode)  # 共享权重,不重新训练
            rollout_rng = random.Random(seed * 13 + (0 if mode == "consensus" else 1))
            for H in H_SWEEP:
                results[mode]["H"][H].append(
                    run_one_setting(view, Vhat, Qstar, K_FIXED_FOR_H_SWEEP, H, rollout_rng, states)
                )
            for K in K_SWEEP:
                results[mode]["K"][K].append(
                    run_one_setting(view, Vhat, Qstar, K, H_FIXED_FOR_K_SWEEP, rollout_rng, states)
                )
        print(f"[seed {seed}] 完成,累计耗时 {time.time() - t0:.1f}s")

    for mode in ("consensus", "member"):
        print(f"\n{'=' * 70}\nsample_mode = {mode}\n{'=' * 70}")
        print(f"扫 H(固定 K={K_FIXED_FOR_H_SWEEP})，{len(SEEDS)} 个种子取均值±标准差")
        print(f"{'H':>3} | {'决策改变率':>14} | {'命中率(改变里)':>16} | {'帮倒忙率(改变里)':>18}")
        summarize(results[mode]["H"], H_SWEEP)

        print(f"\n扫 K(固定 H={H_FIXED_FOR_K_SWEEP})，{len(SEEDS)} 个种子取均值±标准差")
        print(f"{'K':>3} | {'决策改变率':>14} | {'命中率(改变里)':>16} | {'帮倒忙率(改变里)':>18}")
        summarize(results[mode]["K"], K_SWEEP)

    low_hit = [r["low_confidence_hit_rate"] for r in disagreement_results if r["low_confidence_hit_rate"] is not None]
    low_hurt = [r["low_confidence_hurt_rate"] for r in disagreement_results if r["low_confidence_hurt_rate"] is not None]
    high_hit = [r["high_confidence_hit_rate"] for r in disagreement_results if r["high_confidence_hit_rate"] is not None]
    high_hurt = [
        r["high_confidence_hurt_rate"] for r in disagreement_results if r["high_confidence_hurt_rate"] is not None
    ]
    print(f"\n{'=' * 70}\n集成分歧不确定性分组(固定K={UNCERTAINTY_K},H={UNCERTAINTY_H},和sample_mode无关,只算一次)\n{'=' * 70}")
    print(
        f"低置信度(分歧>中位数): 命中率={fmt(statistics.mean(low_hit), statistics.pstdev(low_hit)) if low_hit else 'n/a'}"
        f"  帮倒忙率={fmt(statistics.mean(low_hurt), statistics.pstdev(low_hurt)) if low_hurt else 'n/a'}"
    )
    print(
        f"高置信度(分歧<=中位数): 命中率={fmt(statistics.mean(high_hit), statistics.pstdev(high_hit)) if high_hit else 'n/a'}"
        f"  帮倒忙率={fmt(statistics.mean(high_hurt), statistics.pstdev(high_hurt)) if high_hurt else 'n/a'}"
    )
    print(f"\n总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
