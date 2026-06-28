"""
domain_rand.py — 域随机化 (domain randomization) 的 GPU-free 概念演示 (M11.6).

为什么需要它 (M11.6): sim2real gap —— 在仿真训的策略, 到真实环境会掉点 (sim≠real)。**域随机化
(DR)** 是弥合 gap 的主力: 训练时**随机化仿真参数** (光照/摩擦/传感器噪声/动力学...), 让策略见过
足够多变化, 从而对真实的"未见变化"鲁棒。

本文件用一个 GPU-free 玩具坐实 DR 的核心机制: **域随机化 = 训练时覆盖更广的环境配置**。
这里随机化的「环境参数」= 目标分布 (sim 只在窄区域、real 在全区域):
- 不用 DR: 在**窄**目标区域训 BC → 部署到**全**区域"真实"目标 → 没见过的远目标失败 (sim2real gap)。
- 用 DR:   训练时**随机化目标到全区域** → 策略见过各种配置 → 部署到真实全区域 → 泛化。
策略能观测目标, 所以"覆盖度"是关键: 见过才会做 (同 M11.4 BC 分布漂移, DR = 主动扩训练分布)。

复用 M11.1 toy_env。纯 torch tiny CPU 确定性。(真 IsaacLab 跑见 N2 指引, 需 NV GPU。)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_M11 = Path(__file__).resolve().parents[2] / "embodied-foundations" / "src"
if str(_M11) not in sys.path:
    sys.path.insert(0, str(_M11))
import toy_env as env  # noqa: E402

# 目标区域: 'narrow' = sim 只覆盖右上小块; 'wide' = real 全区域
_REGIONS = {"narrow": (0.3, 0.9, -0.3, 0.3), "wide": (-0.9, 0.9, -0.9, 0.9)}


def _reset_region(rng, region: str):
    """随机起点 + 目标从指定区域采 (域随机化作用在目标分布上)。"""
    pos = rng.uniform(-env.BOX, env.BOX, size=2)
    x0, x1, y0, y1 = _REGIONS[region]
    goal = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], np.float32)
    return np.concatenate([pos, goal]).astype(np.float32)


def collect_demos(n: int = 400, region: str = "narrow", seed: int = 0):
    """采专家 demo, 目标从 region 采。region='wide' = 域随机化 (覆盖全配置)。"""
    rng = np.random.default_rng(seed)
    S, A = [], []
    for _ in range(n):
        s = _reset_region(rng, region)
        for _ in range(env.MAX_T):
            a = env.expert_action(s)
            S.append(s.copy()); A.append(a)
            s = env.step(s, a)
            if env.is_success(s):
                break
    return np.array(S, np.float32), np.array(A, np.float32)


def build_policy(hidden: int = 64, seed: int = 0):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[domain_rand] 无 torch ({exc!r})"); return None
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(env.STATE_DIM, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, env.ACT_DIM))


def train_policy(model, S, A, epochs: int = 300, lr: float = 3e-3, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(S); Y = torch.tensor(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.MSELoss()
    for _ in range(epochs):
        loss = lossf(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def eval_region(model, region: str = "wide", n_episodes: int = 300, seed: int = 100):
    """在某目标区域评估成功率 (region='wide' = 真实全区域)。"""
    import torch
    succ = 0
    for i in range(n_episodes):
        rng = np.random.default_rng(seed + i)
        s = _reset_region(rng, region)
        for _ in range(env.MAX_T):
            a = model(torch.tensor(s[None], dtype=torch.float32)).detach().numpy()[0]
            s = env.step(s, a)
            if env.is_success(s):
                succ += 1; break
    return succ / n_episodes


if __name__ == "__main__":
    print("域随机化 (DR) GPU-free 演示: 目标配置覆盖度 sim2real gap, 环境 = M11.1 2D 到达\n")
    # 不用 DR: 只在窄区域 (sim) 训练
    S0, A0 = collect_demos(n=400, region="narrow", seed=0)
    m0 = build_policy(seed=0); train_policy(m0, S0, A0)
    # 用 DR: 随机化目标到全区域训练
    Sd, Ad = collect_demos(n=400, region="wide", seed=0)
    md = build_policy(seed=0); train_policy(md, Sd, Ad)
    print("在 sim (窄区域目标) 评估:")
    print(f"  无 DR: {eval_region(m0, 'narrow'):.2f}   DR: {eval_region(md, 'narrow'):.2f}")
    print("在 real (全区域目标) 评估:")
    print(f"  无 DR: {eval_region(m0, 'wide'):.2f}   DR: {eval_region(md, 'wide'):.2f}")
    print("→ 无 DR 在窄 sim 好、到全区域真实掉点 (没见过的远目标失败 = sim2real gap);")
    print("  DR 训练覆盖全配置 → 真实全区域泛化。DR = 主动把训练分布扩到覆盖真实。")
