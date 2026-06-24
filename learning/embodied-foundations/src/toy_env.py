"""
toy_env.py — M11 具身模块的共享玩具控制环境 (2D 到达任务, 确定性 CPU).

为什么需要它 (M11.1): 具身 AI 要在「环境-观测-动作」闭环里学。本文件提供一个最小确定性
环境, 作为整个 M11 的共享地基 (BC/扩散策略/世界模型/sim2real 都复用它, 像 M10/M13 的跨专题
src 复用)。

环境: 一个 2D 智能体在 [-1,1] 方盒里, 要走到一个目标点。
  - 状态 state = [agent_x, agent_y, goal_x, goal_y]  (4D, 含目标→策略需泛化到不同目标)
  - 动作 action = [dx, dy]  (2D 连续速度, 会被裁到单位圆 * STEP)
  - 转移 step:  agent += action; 裁在方盒内
  - 专家 expert: 朝目标走 (action = 归一化(goal - agent))
  - 成功 success: 到目标距离 < THRESH

纯 numpy, 确定性 (固定 seed)。
"""
from __future__ import annotations

import numpy as np

BOX = 1.0
STEP = 0.12          # 每步最大位移
THRESH = 0.12        # 到达阈值
MAX_T = 30           # 一个 episode 最多步数
STATE_DIM = 4
ACT_DIM = 2


def reset(rng) -> np.ndarray:
    """随机初始化 agent 与 goal, 返回 state=[ax,ay,gx,gy]。"""
    pos = rng.uniform(-BOX, BOX, size=2)
    goal = rng.uniform(-BOX, BOX, size=2)
    return np.concatenate([pos, goal]).astype(np.float32)


def step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """转移: agent += clip(action) * STEP, 裁方盒。goal 不变。"""
    a = np.asarray(action, dtype=np.float32)
    norm = np.linalg.norm(a)
    if norm > 1.0:
        a = a / norm                       # 裁到单位圆内 (限速)
    pos = state[:2] + a * STEP
    pos = np.clip(pos, -BOX, BOX)
    return np.concatenate([pos, state[2:]]).astype(np.float32)


def expert_action(state: np.ndarray) -> np.ndarray:
    """专家策略: 朝目标走 (归一化方向)。"""
    d = state[2:] - state[:2]
    n = np.linalg.norm(d)
    return (d / n if n > 1e-6 else np.zeros(2)).astype(np.float32)


def is_success(state: np.ndarray) -> bool:
    return bool(np.linalg.norm(state[2:] - state[:2]) < THRESH)


def rollout(policy_fn, seed: int = 0, max_t: int = MAX_T, record: bool = False):
    """用 policy_fn(state)->action 跑一个 episode。返回 (成功?, 步数, [轨迹])。"""
    rng = np.random.default_rng(seed)
    s = reset(rng)
    traj = [s.copy()]
    for t in range(max_t):
        a = policy_fn(s)
        s = step(s, a)
        if record:
            traj.append(s.copy())
        if is_success(s):
            return True, t + 1, (traj if record else None)
    return False, max_t, (traj if record else None)


def make_demos(n: int = 200, seed: int = 0):
    """合成专家 demo: n 个 episode 的 (state, action) 对, 供 behavior cloning。"""
    rng = np.random.default_rng(seed)
    S, A = [], []
    for _ in range(n):
        s = reset(rng)
        for _ in range(MAX_T):
            a = expert_action(s)
            S.append(s.copy()); A.append(a.copy())
            s = step(s, a)
            if is_success(s):
                break
    return np.array(S, np.float32), np.array(A, np.float32)


def eval_policy(policy_fn, n_episodes: int = 100, seed: int = 100) -> float:
    """成功率 = 多个随机 episode 里到达目标的比例。"""
    succ = 0
    for i in range(n_episodes):
        ok, _, _ = rollout(policy_fn, seed=seed + i)
        succ += int(ok)
    return succ / n_episodes


if __name__ == "__main__":
    print(f"环境: 2D 到达, state={STATE_DIM}D action={ACT_DIM}D, 方盒[-{BOX},{BOX}]")
    sr = eval_policy(expert_action, n_episodes=200)
    print(f"专家策略成功率: {sr:.2f} (应接近 1.0)")
    S, A = make_demos(n=100, seed=0)
    print(f"专家 demo: {S.shape[0]} 个 (state,action) 对")
    ok, steps, traj = rollout(expert_action, seed=1, record=True)
    print(f"一个 episode: 成功={ok}, 步数={steps}, 轨迹长度={len(traj)}")
