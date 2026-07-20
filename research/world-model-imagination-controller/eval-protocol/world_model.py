"""LearnedWorldModel:智能体从有限次真实交互采样里,用频次估计学出的不完美转移模型。

这是本 pilot 里"想象"要用的那个模型——它不是真实环境,会犯错,尤其是在采样次数少的
(s,a) 上错得更离谱。visit_count 同时充当"模型置信度"的代理信号,用于后面分析
想象是否在低置信度状态上更有用/更危险。
"""
from __future__ import annotations

import random
from collections import defaultdict

from gridworld_env import ACTIONS, N_STATES, clip_move, is_terminal, rc, state_id, true_transition_dist


def collect_rollout_data(n_transitions: int, rng: random.Random) -> list[tuple[int, str, int]]:
    """behavior policy = 均匀随机动作,在真实环境里跑出有限条 (s,a,s') 转移,给学模型用。

    注意:这里调用 true_transition_dist 是在模拟"智能体和真实世界交互采样",
    不是作弊——智能体只看到采样结果 (s,a,s'),看不到 true_transition_dist 这个函数本身,
    后面 LearnedWorldModel/value_iteration 全部只用这份有限采样数据,不会再碰 true_transition_dist。
    """
    data: list[tuple[int, str, int]] = []
    s = rng.randrange(N_STATES)
    while len(data) < n_transitions:
        if is_terminal(s):
            s = rng.randrange(N_STATES)
            continue
        a = rng.choice(ACTIONS)
        dist = true_transition_dist(s, a)
        states, probs = zip(*dist.items())
        ns = rng.choices(states, weights=probs, k=1)[0]
        data.append((s, a, ns))
        s = ns
    return data


class LearnedWorldModel:
    def __init__(self, data: list[tuple[int, str, int]]):
        self.counts: dict[tuple[int, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for s, a, ns in data:
            self.counts[(s, a)][ns] += 1

    def visit_count(self, s: int, a: str) -> int:
        return sum(self.counts.get((s, a), {}).values())

    def transition_dist(self, s: int, a: str) -> dict[int, float]:
        """精确期望用(baked-in critic 的 value iteration):plain MLE,未见过的 (s,a) 退化为"假设无滑动生效"的乐观先验。"""
        obs = self.counts.get((s, a))
        if not obs:
            r, c = rc(s)
            nr, nc = clip_move(r, c, a)
            return {state_id(nr, nc): 1.0}
        n = sum(obs.values())
        return {k: v / n for k, v in obs.items()}

    def sample_next(self, s: int, a: str, rng: random.Random) -> int:
        """Monte Carlo 想象 rollout 用:按学到的分布采样一个 s'(Laplace +1 平滑,避免小样本下过度自信)。"""
        obs = self.counts.get((s, a))
        if not obs:
            r, c = rc(s)
            nr, nc = clip_move(r, c, a)
            return state_id(nr, nc)
        states = list(obs.keys())
        weights = [obs[k] + 1 for k in states]
        return rng.choices(states, weights=weights, k=1)[0]


if __name__ == "__main__":
    rng = random.Random(0)
    data = collect_rollout_data(400, rng)
    model = LearnedWorldModel(data)
    assert len(data) == 400
    visited = sum(1 for s in range(N_STATES) for a in ACTIONS if model.visit_count(s, a) > 0)
    total_pairs = sum(1 for s in range(N_STATES) if not is_terminal(s)) * len(ACTIONS)
    print(f"400 条转移覆盖了 {visited}/{total_pairs} 个 (s,a) 对(应远小于100%,说明样本确实有限、模型确实不完美)")
    assert visited < total_pairs, "样本覆盖了全部(s,a),模型不会有误差,pilot 的前提假设不成立——需要减少 n_transitions"
    dist = model.transition_dist(0, "RIGHT")
    assert abs(sum(dist.values()) - 1.0) < 1e-9, "transition_dist 没有归一化"
    print("world_model.py 自检通过")
