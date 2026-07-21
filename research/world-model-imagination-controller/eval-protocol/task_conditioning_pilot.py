"""Idea 10 pilot——task-conditioning 扩展:测 Weikai 提的"task conditioned"这一半。

前两轮 pilot(表格模型 RESULTS.md、神经网络集成 RESULTS-neural-ensemble.md)都在测"uncertainty aware"
这一半,而且都得到同一个核心教训:想象如果和基线共享完全相同的信息源,决策时多花算力搜索只会注入噪声,
不会带来真正的改善(member 模式——让想象看到集成分歧这种"廉价"信息优势——也没能扭转这个结论)。

这一轮直接测另一半:如果想象在决策那一刻,真的比基线多知道一件"决策相关"的事——**当前这一局真实的
任务目标是什么**——是不是终于能把"同源想象只有噪声"翻成"想象有真实决策价值"?

**实验设计(三个策略,共享同一个从有限采样学到的不完美动力学模型)**:
- baseline(no-imagination):用 V̂_stale——针对**默认目标**(5,5)算出来的价值函数,代表一个训练/烘焙
  好之后没有针对"今天具体是哪个任务"重新算过的 amortized critic。
- unconditioned imagination(对照组,复刻前两轮的"同源"设计):决策时用同一个不完美动力学模型做 K×H
  Monte Carlo 想象,但续跑策略和 bootstrap 用的还是 V̂_stale——想象没有多知道任何东西,纯粹是"多算几步"。
- task-conditioned imagination(这一轮真正要测的):决策时用 V̂_true——**针对这一局真实目标**算出来的价值
  函数——做续跑和 bootstrap。这是想象唯一"多掌握了什么"的地方:它知道今天的目标是哪个,baseline 不知道。

三者的**即时奖励**在 rollout 过程中都用真实当前目标计算(即时奖励假设总是可观测,这是大多数真实 RL
场景的标准设定;"过时"的只是长期价值估计 V̂,不是眼前一步的奖励信号)。裁判:用真实动力学+真实当前
目标精确算出的 Q*_true。

## 预注册预测(写在跑代码之前)

- **预测A(比较有把握)**:unconditioned imagination vs baseline,命中率应该复现前两轮"明显低于/接近
  50%"的模式,甚至可能更差——因为这次 V̂_stale 不只是"因样本有限而不精确",而是**系统性地假设错了
  目标**,噪声扰动的起点本身就是错的。
- **预测B(核心假设)**:task-conditioned imagination vs baseline,命中率应该**明显超过前两轮任何一次
  实验**,大概率突破 50%——因为这是三轮 pilot 里第一次想象真正拥有 baseline 没有的、和决策直接相关的
  信息(真实目标位置)。这是"给想象真正决策相关信息"这条假设第一次有机会被正面验证,而不是像 member
  模式那样被证伪。
- **预测C(没把握,顺手看一眼)**:V̂_stale 和真实目标"隔得越远"(欧氏距离),task-conditioned 相对
  unconditioned 的命中率提升幅度应该越大——因为目标隔得越远,V̂_stale 错得越离谱,task-conditioning
  能纠正的空间也越大。

**诚实的预期局限**:这个设计某种程度上是"刻意让 baseline 出错"来验证"给想象正确信息能不能纠正"——
这更像是一个存在性验证("genuine信息优势确实能让想象翻正"),不是在测"现实中含噪声的任务条件化信号
能带来多大提升"这个更精细的问题,后者需要更贴近真实场景的后续实验。
"""
from __future__ import annotations

import random
import statistics

from gridworld_env import ACTIONS, GOAL, HAZARDS, N_STATES, is_terminal, rc, value_iteration, true_transition_dist
from world_model import LearnedWorldModel, collect_rollout_data
from imagination_planner import imagine_action, no_imagination_action

N_TRAIN_TRANSITIONS = 400
SEEDS = [0, 1, 2, 3, 4]
K, H = 10, 5  # 固定一个较慷慨的预算,这一轮的自变量是"信息来源"不是budget本身
CANDIDATE_GOALS = [(0, 0), (0, 5), (5, 0)]  # 3个角落,都远离默认目标(5,5),且不与HAZARDS重叠


def _state_id(pos: tuple[int, int]) -> int:
    r, c = pos
    return r * 6 + c


def eval_states(true_goal: tuple[int, int]) -> list[int]:
    """排除对"今天这局"而言的终止态(命中真实目标或hazard),不是排除默认GOAL。"""
    return [s for s in range(N_STATES) if not is_terminal(s, true_goal)]


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def run_condition(model, Vhat_baseline, Vhat_imagine, Qstar_true, true_goal, rng, states) -> dict:
    """a0(baseline)永远用 Vhat_baseline(=Vhat_stale)算,a1(想象)用 Vhat_imagine 算——
    这样"unconditioned想象"(Vhat_imagine=Vhat_stale)和"task-conditioned想象"(Vhat_imagine=Vhat_true)
    才是在跟**同一个**baseline比,唯一变量是想象自己用的信息源,不会把baseline也悄悄换掉。
    """
    changed = hit = hurt = 0
    for s in states:
        a0 = no_imagination_action(model, Vhat_baseline, s, goal=true_goal)
        a1 = imagine_action(model, Vhat_imagine, s, K, H, rng, goal=true_goal)
        if a1 != a0:
            changed += 1
            if Qstar_true[(s, a1)] > Qstar_true[(s, a0)] + 1e-9:
                hit += 1
            elif Qstar_true[(s, a1)] < Qstar_true[(s, a0)] - 1e-9:
                hurt += 1
    n = len(states)
    return {
        "decision_change_rate": changed / n,
        "hit_rate": (hit / changed) if changed else None,
        "hurt_rate": (hurt / changed) if changed else None,
        "n": n,
    }


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def main() -> None:
    assert GOAL not in CANDIDATE_GOALS, "候选目标不该和默认GOAL重合,否则测不出mismatch"
    for g in CANDIDATE_GOALS:
        assert g not in HAZARDS, f"候选目标 {g} 和 HAZARDS 重叠,设计有误"

    results_uncond = {g: [] for g in CANDIDATE_GOALS}
    results_taskcond = {g: [] for g in CANDIDATE_GOALS}

    for seed in SEEDS:
        rng = random.Random(seed)
        data = collect_rollout_data(N_TRAIN_TRANSITIONS, rng)
        model = LearnedWorldModel(data)
        Vhat_stale, _ = value_iteration(model.transition_dist, goal=GOAL)

        for true_goal in CANDIDATE_GOALS:
            Vhat_true, _ = value_iteration(model.transition_dist, goal=true_goal)
            _, Qstar_true = value_iteration(true_transition_dist, goal=true_goal)
            states = eval_states(true_goal)

            r_uncond = run_condition(model, Vhat_stale, Vhat_stale, Qstar_true, true_goal, rng, states)
            r_taskcond = run_condition(model, Vhat_stale, Vhat_true, Qstar_true, true_goal, rng, states)
            results_uncond[true_goal].append(r_uncond)
            results_taskcond[true_goal].append(r_taskcond)

        print(f"[seed {seed}] 完成")

    print(f"\n{'=' * 78}")
    print("按候选目标分组(K={}, H={}, {} 个种子)".format(K, H, len(SEEDS)))
    print(f"{'=' * 78}")
    print(f"{'目标':>10} | {'与默认目标距离':>10} | {'模式':>22} | {'决策改变率':>14} | {'命中率':>14} | {'帮倒忙率':>14}")
    for g in CANDIDATE_GOALS:
        dist = euclidean(g, GOAL)
        ru, rt = results_uncond[g], results_taskcond[g]
        print(f"{str(g):>10} | {dist:>10.2f} | {'unconditioned想象':>22} | "
              f"{fmt([r['decision_change_rate'] for r in ru]):>14} | {fmt([r['hit_rate'] for r in ru]):>14} | "
              f"{fmt([r['hurt_rate'] for r in ru]):>14}")
        print(f"{'':>10} | {'':>10} | {'task-conditioned想象':>22} | "
              f"{fmt([r['decision_change_rate'] for r in rt]):>14} | {fmt([r['hit_rate'] for r in rt]):>14} | "
              f"{fmt([r['hurt_rate'] for r in rt]):>14}")

    print(f"\n{'=' * 78}\n汇总(跨3个候选目标合并,{len(SEEDS)}个种子)\n{'=' * 78}")
    all_uncond_hit = [r["hit_rate"] for g in CANDIDATE_GOALS for r in results_uncond[g]]
    all_taskcond_hit = [r["hit_rate"] for g in CANDIDATE_GOALS for r in results_taskcond[g]]
    print(f"unconditioned想象   命中率 = {fmt(all_uncond_hit)}")
    print(f"task-conditioned想象 命中率 = {fmt(all_taskcond_hit)}")


if __name__ == "__main__":
    main()
