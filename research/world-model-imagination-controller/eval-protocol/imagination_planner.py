"""想象规划器:H=0(不想象,信任已经烘焙好的价值函数)baseline + 固定预算(K,H) Monte Carlo 想象 baseline。

两者共享同一个不完美的 LearnedWorldModel 和同一个 baked-in 价值函数 Vhat——
唯一区别是"想象"版本会在决策那一刻,用模型多采样几次、往前多看几步再决定,
而不是像 H=0 版本一样直接信任已经算好的 Vhat。这样"想象是否有用"这个问题
就干净地归结为:对同一个不完美模型多花算力重新搜索,是修正了错误,还是只是
把同一个模型的噪声/偏差重新采样了一遍、白费算力甚至帮倒忙。
"""
from __future__ import annotations

import random

from gridworld_env import ACTIONS, GAMMA, GOAL, is_terminal, reward
from world_model import LearnedWorldModel


def no_imagination_action(model: LearnedWorldModel, Vhat: list[float], s: int, goal: tuple = GOAL) -> str:
    """H=0 基线:只做一步精确期望展开(用 Vhat 当 bootstrap),不做决策时的多步 Monte Carlo 搜索。

    goal 参数(默认 GOAL,向后兼容)只影响这一步的**即时奖励**怎么算——即时奖励假设总是可观测的
    (符合大多数真实RL设定),"过时"的只可能是 Vhat 本身(调用方决定传入哪个 goal 算出来的 Vhat)。
    """
    best_a, best_val = None, float("-inf")
    for a in ACTIONS:
        dist = model.transition_dist(s, a)
        q = sum(p * (reward(ns, goal) + GAMMA * Vhat[ns]) for ns, p in dist.items())
        if q > best_val:
            best_val, best_a = q, a
    return best_a


def _simulate_rollout(
    model: LearnedWorldModel, Vhat: list[float], s: int, first_action: str, horizon: int, rng: random.Random,
    goal: tuple = GOAL,
) -> float:
    """从 s 执行 first_action,续跑策略=信任 Vhat 的贪心策略,跑满 horizon 步后用 Vhat 做 bootstrap(仿 TD-MPC 短rollout+终值)。"""
    total, disc = 0.0, 1.0
    cur, a = s, first_action
    for _ in range(horizon):
        ns = model.sample_next(cur, a, rng)  # 真·Monte Carlo 采样,不是精确期望
        total += disc * reward(ns, goal)
        disc *= GAMMA
        cur = ns
        if is_terminal(cur, goal):
            return total
        a = no_imagination_action(model, Vhat, cur, goal)
    total += disc * Vhat[cur]
    return total


def imagine_action(
    model: LearnedWorldModel, Vhat: list[float], s: int, K: int, H: int, rng: random.Random, goal: tuple = GOAL,
) -> str:
    """固定预算 (K,H):每个候选首动作跑 K 条深度 H 的想象 rollout,取均值回报最高的动作。"""
    best_a, best_val = None, float("-inf")
    for a in ACTIONS:
        vals = [_simulate_rollout(model, Vhat, s, a, H, rng, goal) for _ in range(K)]
        v = sum(vals) / len(vals)
        if v > best_val:
            best_val, best_a = v, a
    return best_a


if __name__ == "__main__":
    from gridworld_env import N_STATES, value_iteration
    from world_model import collect_rollout_data

    rng = random.Random(0)
    data = collect_rollout_data(400, rng)
    model = LearnedWorldModel(data)
    Vhat, _ = value_iteration(model.transition_dist)

    s0 = 0  # 左上角
    a_no_imag = no_imagination_action(model, Vhat, s0)
    a_imag = imagine_action(model, Vhat, s0, K=10, H=5, rng=rng)
    print(f"state=0(左上角): no-imagination 选 {a_no_imag}, imagination(K=10,H=5) 选 {a_imag}")
    assert a_no_imag in ACTIONS and a_imag in ACTIONS

    # sanity: K=1,H=1 时想象 rollout 应该退化到和"一步展开"同量级(不应该系统性跑出天差地别的值)
    one_step_val = max(
        sum(p * (reward(ns) + GAMMA * Vhat[ns]) for ns, p in model.transition_dist(s0, a).items())
        for a in ACTIONS
    )
    tiny_imag_val = max(
        sum(_simulate_rollout(model, Vhat, s0, a, 1, rng) for _ in range(50)) / 50 for a in ACTIONS
    )
    assert abs(one_step_val - tiny_imag_val) < 0.15, (
        f"K大H=1时想象值({tiny_imag_val:.3f})应该在一步精确展开值({one_step_val:.3f})附近,偏差过大说明有bug"
    )
    print(f"sanity check 通过:精确一步展开={one_step_val:.3f}, 大样本H=1想象均值={tiny_imag_val:.3f}(应接近)")
