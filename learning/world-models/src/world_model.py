"""
world_model.py — 动作条件世界模型 (玩具环境), 学「状态+动作→下一状态」, 然后脑内想象 rollout.

为什么需要它 (M13.5): 视频生成 (M13.4) 再进一步 —— 如果模型能预测「我做某动作后世界变成
什么样」, 它就是**世界模型 (world model)**: 学环境的转移动态。学会后能在脑内**想象** (imagine)
未来, 不碰真环境就规划/做 model-based RL。这也是「视频模型当模拟器」的内核 (OpenAI 2024 洞察),
并直通 M11 具身: 机器人用世界模型预演动作后果。

本文件玩具环境: 一个 2D 智能体在 [-1,1] 方盒里, 动作 = {上下左右} 各推一步 (带阻尼+轻噪),
撞墙夹住。世界模型 (MLP) 学 (state, action)→Δstate。学会后:
  - imagine(): 给起点+一串动作, 链式预测整条轨迹 (脑内 rollout, 不碰真环境)
  - 多步误差: 1 步准 ≠ 多步准, 误差复利式累积 (世界模型核心评测)

纯 torch (tiny CPU), 确定性。与 M11 world-action-models 同源/共享。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 动作 → 单位推力 (上/下/左/右)
ACTIONS = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
N_ACTIONS = len(ACTIONS)
STEP = 0.15        # 每步推力大小
DAMP = 0.0         # (保留) 阻尼
BOX = 1.0          # 方盒边界


def true_step(state: np.ndarray, action: int, rng=None) -> np.ndarray:
    """真环境一步: state + 动作推力 (+轻噪), 夹在方盒内。state:(...,2)。"""
    push = ACTIONS[action] * STEP
    nxt = state + push
    if rng is not None:
        nxt = nxt + 0.01 * rng.standard_normal(nxt.shape).astype(np.float32)
    return np.clip(nxt, -BOX, BOX).astype(np.float32)


def make_transitions(n: int = 4000, seed: int = 0):
    """随机采 (state, action) 转移样本, 用于训练世界模型。返回 (states, actions, deltas)。"""
    rng = np.random.default_rng(seed)
    states = rng.uniform(-BOX, BOX, size=(n, 2)).astype(np.float32)
    actions = rng.integers(0, N_ACTIONS, size=n)
    nxt = np.stack([true_step(states[i], int(actions[i]), rng) for i in range(n)])
    deltas = (nxt - states).astype(np.float32)        # 预测「变化量」更稳
    return states, actions, deltas


def build_world_model(hidden: int = 64, seed: int = 0):
    """世界模型: (state, action_onehot) → Δstate 的 MLP。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[world_model] 无 torch ({exc!r})")
        return None
    torch.manual_seed(seed)

    class WM(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 + N_ACTIONS, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2))

        def forward(self, s, a_onehot):
            import torch
            return self.net(torch.cat([s, a_onehot], -1))

    return WM()


def _onehot(actions, torch):
    a = torch.tensor(np.asarray(actions))
    return torch.nn.functional.one_hot(a, N_ACTIONS).float()


def train_world_model(model, data, epochs: int = 400, lr: float = 3e-3, seed: int = 0):
    """训练世界模型预测 Δstate。返回 loss 历史。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    states, actions, deltas = data
    s = torch.tensor(states); a = _onehot(actions, torch); d = torch.tensor(deltas)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        pred = model(s, a)
        loss = lossf(pred, d)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def imagine(model, start: np.ndarray, actions, ) -> np.ndarray:
    """脑内 rollout: 用世界模型链式预测整条轨迹 (不碰真环境)。返回 (T+1, 2)。"""
    import torch
    s = np.asarray(start, dtype=np.float32).copy()
    traj = [s.copy()]
    with torch.no_grad():
        for act in actions:
            so = torch.tensor(s[None]); ao = _onehot([act], torch)
            delta = model(so, ao).numpy()[0]
            s = np.clip(s + delta, -BOX, BOX).astype(np.float32)
            traj.append(s.copy())
    return np.array(traj, dtype=np.float32)


def true_rollout(start: np.ndarray, actions, seed: int = 0) -> np.ndarray:
    """真环境 rollout (同一串动作), 当作对照。返回 (T+1, 2)。"""
    rng = np.random.default_rng(seed)
    s = np.asarray(start, dtype=np.float32).copy()
    traj = [s.copy()]
    for act in actions:
        s = true_step(s, int(act), rng)
        traj.append(s.copy())
    return np.array(traj, dtype=np.float32)


def multistep_error(model, n_traj: int = 50, horizon: int = 20, seed: int = 0):
    """多步预测误差 vs 步数: 想象 rollout 与真 rollout 的逐步距离 (误差累积)。"""
    rng = np.random.default_rng(seed)
    errs = np.zeros(horizon + 1)
    for _ in range(n_traj):
        start = rng.uniform(-BOX, BOX, size=2).astype(np.float32)
        acts = rng.integers(0, N_ACTIONS, size=horizon)
        img = imagine(model, start, acts)
        tru = true_rollout(start, acts, seed=int(rng.integers(1e6)))
        errs += np.linalg.norm(img - tru, axis=-1)
    return errs / n_traj          # (horizon+1,) 平均逐步误差


if __name__ == "__main__":
    data = make_transitions(n=4000, seed=0)
    print(f"转移样本: states{data[0].shape} actions{data[1].shape}")
    model = build_world_model()
    if model is not None:
        losses = train_world_model(model, data, epochs=400)
        print(f"世界模型训练: loss {losses[0]:.4f} → {losses[-1]:.4f}")
        # 想象 vs 真实
        start = np.array([0.0, 0.0], np.float32)
        acts = [3, 3, 0, 0, 2, 0, 3]      # 右右上上左上右
        img = imagine(model, start, acts)
        tru = true_rollout(start, acts, seed=1)
        print(f"想象终点 {img[-1].round(3)}  真实终点 {tru[-1].round(3)}  差 {np.linalg.norm(img[-1]-tru[-1]):.3f}")
        errs = multistep_error(model, n_traj=50, horizon=20)
        print(f"多步误差: 1步 {errs[1]:.3f} → 10步 {errs[10]:.3f} → 20步 {errs[20]:.3f} (复利累积)")
        print("→ 世界模型学会环境动态, 能脑内想象 rollout; 但误差随步数累积 (世界模型核心难题)。")
