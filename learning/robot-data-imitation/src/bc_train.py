"""
bc_train.py — 行为克隆 (behavior cloning) + 数据 scaling (M11.4).

为什么需要它 (M11.4): 动作头怎么训? 最基础的答案是**模仿学习 / 行为克隆 (BC)**: 从专家
demo 学策略 = 监督学习 (state → action)。机器人基础模型的训练大头就是大规模 BC。本专题讲 BC
的原理、它的经典毛病 (分布漂移/复合误差), 以及**数据 scaling** (更多 demo → 更好策略)。

复用 M11.1 的 toy_env (2D 到达任务) 当环境与专家。BC = 回归 state→action。纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 复用 M11.1 的共享环境
_M11 = Path(__file__).resolve().parents[2] / "embodied-foundations" / "src"
if str(_M11) not in sys.path:
    sys.path.insert(0, str(_M11))
import toy_env as env  # noqa: E402


def build_bc_policy(hidden: int = 64, seed: int = 0):
    """BC 策略: state → action 的回归 MLP (监督模仿专家)。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[bc_train] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(env.STATE_DIM, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, env.ACT_DIM))


def train_bc(model, S: np.ndarray, A: np.ndarray, epochs: int = 300, lr: float = 3e-3, seed: int = 0):
    """行为克隆 = 监督回归 state→action。返回 loss 历史。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(S); Y = torch.tensor(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        loss = lossf(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def bc_policy_fn(model):
    import torch

    def fn(state):
        return model(torch.tensor(np.asarray(state, np.float32)[None])).detach().numpy()[0]
    return fn


def train_from_demos(n_demos: int, epochs: int = 300, seed: int = 0):
    """采 n_demos 条专家 demo → 训 BC 策略 → 返回 (model, 成功率)。"""
    S, A = env.make_demos(n=n_demos, seed=seed)
    model = build_bc_policy(seed=seed)
    if model is None:
        return None, 0.0
    train_bc(model, S, A, epochs=epochs, seed=seed)
    sr = env.eval_policy(bc_policy_fn(model), n_episodes=200)
    return model, sr


def scaling_curve(demo_sizes=(2, 5, 10, 25, 50, 100, 200), epochs: int = 300, seed: int = 0):
    """数据 scaling: 不同 demo 数量 vs 成功率。返回 (sizes, success_rates, n_pairs)。"""
    srs, npairs = [], []
    for n in demo_sizes:
        S, A = env.make_demos(n=n, seed=seed)
        model = build_bc_policy(seed=seed)
        train_bc(model, S, A, epochs=epochs, seed=seed)
        srs.append(env.eval_policy(bc_policy_fn(model), n_episodes=200))
        npairs.append(len(S))
    return list(demo_sizes), srs, npairs


if __name__ == "__main__":
    print("行为克隆 (BC) + 数据 scaling, 环境 = M11.1 2D 到达任务\n")
    for n in [3, 10, 50, 200]:
        _, sr = train_from_demos(n_demos=n, epochs=300)
        print(f"  {n:3d} 条 demo → BC 成功率 {sr:.2f}")
    print("\n→ 更多 demo → 更高成功率 (数据 scaling); 太少 demo 覆盖不到的状态会失败 (分布漂移)。")
