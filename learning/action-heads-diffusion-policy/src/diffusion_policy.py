"""
diffusion_policy.py — 扩散动作头 (diffusion policy), 解决多峰动作 (M11.3).

为什么需要它 (M11.3): M11.2-L3 揭示矛盾 —— 连续回归动作头**怕多峰** (MSE 取平均), 离散 token
精度粗。机器人动作经常多峰 (绕障碍可上可下)。**扩散动作头** (你的 M13!) 同时解决: 既连续平滑、
又能表达多峰。这是 π 系列的核心。

本文件玩具: 绕障碍导航。起点在左、目标在右、中间一个圆障碍。专家 50% 从上绕、50% 从下绕
(→ 在障碍附近动作是**双峰**: 朝上或朝下)。对比:
  - 连续回归: 学成"取平均" → 直冲障碍 → 撞 (多峰死穴)
  - 扩散策略: 采样到某一峰 (上或下) → 绕开 → 成功

扩散机制 = M13.1 的 DDPM (前向加噪/反向去噪预测噪声), **加上状态条件** + 可选 action chunking
(一次预测 H 步动作)。与 M13 `diffusion.py` 同源 (条件版)。纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 环境: 绕障碍导航
START_X = -0.85
GOAL = np.array([0.9, 0.0], dtype=np.float32)
OBST_C = np.array([0.0, 0.0], dtype=np.float32)
OBST_R = 0.35
STEP = 0.12
THRESH = 0.14
MAX_T = 40
ACT_DIM = 2
STATE_DIM = 2


def collides(pos) -> bool:
    return bool(np.linalg.norm(np.asarray(pos)[:2] - OBST_C) < OBST_R)


def reached(pos) -> bool:
    return bool(np.linalg.norm(np.asarray(pos)[:2] - GOAL) < THRESH)


def _expert_target(pos, side):
    """专家两段路: 先去障碍上/下方的 waypoint, 再去目标。side=+1 上, -1 下。"""
    waypoint = np.array([0.0, side * (OBST_R + 0.25)], np.float32)
    if pos[0] < -0.05 and abs(pos[1] - waypoint[1]) > 0.12:
        return waypoint
    return GOAL


def make_obstacle_demos(n: int = 300, chunk: int = 1, seed: int = 0):
    """合成双峰专家 demo: 一半从上绕、一半从下绕。返回 (states (N,2), action_chunks (N, chunk*2))。"""
    rng = np.random.default_rng(seed)
    S, A = [], []
    for i in range(n):
        side = 1.0 if i % 2 == 0 else -1.0           # 双峰: 交替上/下
        pos = np.array([START_X, rng.uniform(-0.15, 0.15)], np.float32)
        traj_states, traj_acts = [], []
        for _ in range(MAX_T):
            tgt = _expert_target(pos, side)
            d = tgt - pos; nrm = np.linalg.norm(d)
            a = (d / nrm if nrm > 1e-6 else np.zeros(2)).astype(np.float32)
            traj_states.append(pos.copy()); traj_acts.append(a.copy())
            pos = pos + a * STEP
            if reached(pos):
                break
        # 切成 (state, 接下来 chunk 步动作) 对
        for t in range(len(traj_acts)):
            chunk_a = traj_acts[t:t + chunk]
            while len(chunk_a) < chunk:
                chunk_a.append(chunk_a[-1])          # 末尾不足 chunk 则重复最后一个
            S.append(traj_states[t]); A.append(np.concatenate(chunk_a))
    return np.array(S, np.float32), np.array(A, np.float32)


def make_beta_schedule(T: int = 50):
    betas = np.linspace(1e-4, 0.2, T).astype(np.float32)
    return betas, (1 - betas), np.cumprod(1 - betas).astype(np.float32)


# ───────────────── 扩散动作头 (条件 DDPM, 同 M13.1 + 状态条件) ─────────────────
def build_diffusion_policy(chunk: int = 1, hidden: int = 96, seed: int = 0):
    """条件去噪器: (噪声动作块, 状态, t) → 预测噪声。动作块维度 = chunk*ACT_DIM。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[diffusion_policy] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)
    adim = chunk * ACT_DIM

    class CondDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk = chunk
            self.net = nn.Sequential(
                nn.Linear(adim + STATE_DIM + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, adim))

        def forward(self, a_noisy, state, t):       # a_noisy:(B,adim) state:(B,2) t:(B,1)
            import torch
            return self.net(torch.cat([a_noisy, state, t], -1))

    return CondDenoiser()


def train_diffusion_policy(model, S, A, T: int = 50, epochs: int = 600, lr: float = 2e-3, seed: int = 0):
    """训练条件扩散动作头 (预测噪声, 条件于状态)。返回 (losses, schedule)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    betas, alphas, abars = make_beta_schedule(T)
    s = torch.tensor(S); a = torch.tensor(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        t = rng.integers(0, T, size=len(A))
        ab = abars[t][:, None]
        eps = rng.standard_normal(A.shape).astype(np.float32)
        at = (np.sqrt(ab) * A + np.sqrt(1 - ab) * eps).astype(np.float32)
        pred = model(torch.tensor(at), s, torch.tensor((t[:, None] / T).astype(np.float32)))
        loss = lossf(pred, torch.tensor(eps))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, (betas, alphas, abars)


def sample_action(model, state, schedule, seed: int = 0):
    """反向去噪生成一个动作块 (条件于状态)。返回 (chunk, ACT_DIM)。"""
    import torch
    betas, alphas, abars = schedule
    T = len(betas); rng = np.random.default_rng(seed)
    adim = model.chunk * ACT_DIM
    s = torch.tensor(np.asarray(state, np.float32)[None])
    x = rng.standard_normal((1, adim)).astype(np.float32)
    with torch.no_grad():
        for t in reversed(range(T)):
            tn = torch.full((1, 1), t / T)
            eps = model(torch.tensor(x), s, tn).numpy()
            ab = abars[t]; al = alphas[t]; b = betas[t]
            mean = (1 / np.sqrt(al)) * (x - (b / np.sqrt(1 - ab)) * eps)
            x = mean + (np.sqrt(b) * rng.standard_normal((1, adim)).astype(np.float32) if t > 0 else 0)
    return x.reshape(model.chunk, ACT_DIM)


# ───────────────── 连续回归动作头 (对照, 会取平均) ─────────────────
def build_regression(chunk: int = 1, hidden: int = 96, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    adim = chunk * ACT_DIM

    class Reg(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk = chunk
            self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, adim))

        def forward(self, s):
            return self.net(s)

    return Reg()


def train_regression(model, S, A, epochs: int = 600, lr: float = 2e-3, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    s = torch.tensor(S); a = torch.tensor(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    for _ in range(epochs):
        loss = lossf(model(s), a)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


# ───────────────── rollout / 评测 ─────────────────
def _norm(a):
    n = np.linalg.norm(a)
    return a / n if n > 1.0 else a


def rollout(action_fn, seed: int = 0, record: bool = False):
    """用 action_fn(state)->(chunk,2) 跑一个 episode (open-loop 执行整块再重规划)。
    返回 (success, collided, steps, [traj])。"""
    rng = np.random.default_rng(seed)
    pos = np.array([START_X, rng.uniform(-0.15, 0.15)], np.float32)
    traj = [pos.copy()]; steps = 0
    while steps < MAX_T:
        chunk = np.atleast_2d(action_fn(pos))
        for a in chunk:
            pos = pos + _norm(np.asarray(a, np.float32)) * STEP
            traj.append(pos.copy()); steps += 1
            if collides(pos):
                return False, True, steps, (traj if record else None)
            if reached(pos):
                return True, False, steps, (traj if record else None)
            if steps >= MAX_T:
                break
    return False, False, steps, (traj if record else None)


def eval_policy(action_fn, n_episodes: int = 100, seed: int = 100):
    succ = coll = 0
    for i in range(n_episodes):
        ok, c, _, _ = rollout(action_fn, seed=seed + i)
        succ += int(ok); coll += int(c)
    return succ / n_episodes, coll / n_episodes


def make_diffusion_action_fn(model, schedule, base_seed: int = 0):
    """把扩散动作头包成 action_fn(state)->(chunk,2)。确定性计数器种子 (不用 hash, 避 PYTHONHASHSEED 坑)。"""
    counter = {"n": 0}

    def fn(state):
        counter["n"] += 1
        return sample_action(model, state, schedule, seed=base_seed * 100000 + counter["n"])
    return fn


def make_regression_action_fn(model):
    """把回归动作头包成 action_fn(state)->(chunk,2)。"""
    import torch

    def fn(state):
        out = model(torch.tensor(np.asarray(state, np.float32)[None])).detach().numpy()[0]
        return out.reshape(model.chunk, ACT_DIM)
    return fn


if __name__ == "__main__":
    import torch
    S, A = make_obstacle_demos(n=300, chunk=1, seed=0)
    print(f"双峰专家 demo: {S.shape[0]} 个 (state, action) 对")

    # 扩散动作头
    dp = build_diffusion_policy(chunk=1)
    _, sched = train_diffusion_policy(dp, S, A, epochs=500)
    sr_d, cr_d = eval_policy(make_diffusion_action_fn(dp, sched), n_episodes=100)

    # 连续回归对照
    reg = build_regression(chunk=1); train_regression(reg, S, A, epochs=500)
    sr_r, cr_r = eval_policy(make_regression_action_fn(reg), n_episodes=100)

    print(f"扩散动作头:  成功率 {sr_d:.2f}, 撞障率 {cr_d:.2f}")
    print(f"连续回归:    成功率 {sr_r:.2f}, 撞障率 {cr_r:.2f}")
    print("→ 扩散动作头采样到某一峰 (上/下绕) 避开障碍; 连续回归取平均直冲障碍 (多峰死穴)。")
