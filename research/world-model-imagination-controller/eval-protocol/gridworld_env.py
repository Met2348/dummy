"""SlipperyGridWorld:一个真实转移函数完全已知的 6x6 随机格子世界。

用于给 idea 10(诊断性研究)的 pilot 提供一个"ground truth oracle"——真实转移函数已知,
所以可以精确算出最优 Q*(s,a),用它当裁判去判断"imagination 选的动作到底是不是真的更好"。
智能体自己看不到这个真实转移函数,只能从有限次采样里学一个不完美的模型(见 world_model.py)。
"""
from __future__ import annotations

SIZE = 6
N_STATES = SIZE * SIZE
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

GOAL = (5, 5)
HAZARDS = {(2, 2), (3, 4), (1, 4)}
STEP_COST = -0.02
GOAL_REWARD = 1.0
HAZARD_REWARD = -1.0
SLIP_PROB = 0.25  # 意图动作以 (1-SLIP_PROB) 概率生效,否则等概率替换成4个动作里随机一个
GAMMA = 0.95


def rc(s: int) -> tuple[int, int]:
    return divmod(s, SIZE)


def state_id(r: int, c: int) -> int:
    return r * SIZE + c


def clip_move(r: int, c: int, action: str) -> tuple[int, int]:
    dr, dc = DELTA[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < SIZE):
        nr = r
    if not (0 <= nc < SIZE):
        nc = c
    return nr, nc


def is_terminal(s: int, goal: tuple[int, int] = GOAL) -> bool:
    """goal 可覆盖:task-conditioning 实验里,"当前任务的目标格子"可以和默认 GOAL 不同。"""
    pos = rc(s)
    return pos == goal or pos in HAZARDS


def reward(s_next: int, goal: tuple[int, int] = GOAL) -> float:
    """奖励在"进入 s_next 的那一刻"发放(标准 absorbing-terminal 记法)。goal 可覆盖,见 is_terminal。"""
    pos = rc(s_next)
    if pos == goal:
        return GOAL_REWARD
    if pos in HAZARDS:
        return HAZARD_REWARD
    return STEP_COST


def true_transition_dist(s: int, action: str, goal: tuple[int, int] = GOAL) -> dict[int, float]:
    """真实环境的 P(s'|s,action)——只用来(a)生成训练数据 (b)算 ground-truth Q*,智能体的模型看不到这个函数本身。

    注意:滑动动力学本身和 goal 无关(挪到哪格纯粹是物理),goal 只影响"是否已经到终点"这一件事。
    """
    if is_terminal(s, goal):
        return {s: 1.0}
    r, c = rc(s)
    dist: dict[int, float] = {}
    for actual in ACTIONS:
        p = (1 - SLIP_PROB) + SLIP_PROB / 4 if actual == action else SLIP_PROB / 4
        nr, nc = clip_move(r, c, actual)
        ns = state_id(nr, nc)
        dist[ns] = dist.get(ns, 0.0) + p
    return dist


def value_iteration(transition_fn, goal: tuple[int, int] = GOAL, theta: float = 1e-9, max_iters: int = 10000):
    """通用 value iteration,transition_fn(s,a)->{s':p} 既可以传真实转移(算ground truth)也可以传学到的模型
    (算baked-in critic)。goal 参数让同一个 transition_fn 可以针对不同"当前任务目标"分别算价值函数——
    task-conditioning 实验用这个来对比"baseline 用默认 goal 算的 V̂"和"想象用真实 goal 算的 V̂"。
    """
    V = [0.0] * N_STATES
    for it in range(max_iters):
        delta = 0.0
        newV = V[:]
        for s in range(N_STATES):
            if is_terminal(s, goal):
                newV[s] = 0.0
                continue
            best = float("-inf")
            for a in ACTIONS:
                dist = transition_fn(s, a)
                q = sum(p * (reward(ns, goal) + GAMMA * V[ns]) for ns, p in dist.items())
                best = max(best, q)
            newV[s] = best
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < theta:
            break
    else:
        raise RuntimeError(f"value iteration 在 {max_iters} 轮内未收敛(delta={delta})")

    Q: dict[tuple[int, str], float] = {}
    for s in range(N_STATES):
        for a in ACTIONS:
            dist = transition_fn(s, a)
            Q[(s, a)] = sum(p * (reward(ns, goal) + GAMMA * V[ns]) for ns, p in dist.items())
    return V, Q


if __name__ == "__main__":
    V, Q = value_iteration(true_transition_dist)
    assert all(is_terminal(s) or V[s] > -1.0 for s in range(N_STATES)), "V 值域异常,检查 Bellman 更新"
    non_terminal = [s for s in range(N_STATES) if not is_terminal(s)]
    print(f"真实环境 value iteration 收敛,{len(non_terminal)} 个非终止态,V 范围 "
          f"[{min(V[s] for s in non_terminal):.3f}, {max(V[s] for s in non_terminal):.3f}]")
    goal_adjacent = state_id(5, 4)
    best_a = max(ACTIONS, key=lambda a: Q[(goal_adjacent, a)])
    print(f"sanity check:goal(5,5)正上方/左边格子 state={goal_adjacent} 的最优动作={best_a}(期望是 RIGHT 或 DOWN 靠近目标)")
    assert best_a in ("RIGHT", "DOWN"), f"最优动作不合理:{best_a},说明value iteration或环境定义有bug"
