"""
flow_matching.py — flow matching / rectified flow, 攻 DDPM 的「T 步慢」, 用更少步生成.

为什么需要它 (M13.2): M13.1 的 DDPM 要迭代 T (几十到上千) 步采样, 慢。flow matching 换个视角:
不学「去噪」, 而学一个**速度场 (velocity field)** v_θ(x,t), 它告诉每个点「往哪个方向、多快地
流向数据」。生成 = 从噪声出发, 沿速度场积分一条 ODE 轨迹到数据。如果路径被「拉直」
(rectified flow), 就能用极少步 (甚至 1 步) 走完 —— 这是 2024-26 的 SOTA 采样范式。

本文件在 2D 玩具分布上实现:
  - 训练: 速度场匹配「噪声→数据」直线路径的速度 (conditional flow matching)
  - 采样: ODE Euler 积分 (可指定步数), 看少步也能生成
  - 对比 DDPM: 同样的玩具数据, flow matching 用更少步达到同等质量

纯 torch (tiny CPU), 确定性。你 EE 的 ODE/向量场直觉在这里是优势。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_two_moons(n: int = 512, noise: float = 0.08, seed: int = 0) -> np.ndarray:
    """2D 双月 (同 M13.1, 便于和 DDPM 对比)。"""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = np.pi * rng.random(n1)
    m1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    t2 = np.pi * rng.random(n - n1)
    m2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], 1)
    x = np.concatenate([m1, m2], 0) + noise * rng.standard_normal((n, 2))
    return ((x - x.mean(0)) / x.std(0)).astype(np.float32)


def build_velocity_field(dim: int = 2, hidden: int = 64, seed: int = 0):
    """速度场 v_θ(x, t): 一个 MLP, 输入当前点 + 时间, 输出速度向量. 无 torch 返回 None。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[flow_matching] 无 torch ({exc!r})")
        return None
    torch.manual_seed(seed)

    class Velocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, dim))

        def forward(self, x, t):  # x:(B,dim), t:(B,1)
            import torch
            return self.net(torch.cat([x, t], -1))

    return Velocity()


def train_flow_matching(model, x1: np.ndarray, epochs: int = 500, lr: float = 2e-3,
                        seed: int = 0):
    """训练速度场 (conditional flow matching, 直线路径).

    思想: 对每个数据点 x1, 配一个随机噪声 x0; 它们之间走直线 x_t = (1-t)x0 + t·x1。
    这条直线的速度恒为 (x1 - x0)。训练 v_θ(x_t, t) 去匹配这个速度。
    生成时沿学到的速度场积分, 就能从噪声流到数据。
    """
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    X1 = x1
    losses = []
    for _ in range(epochs):
        x0 = rng.standard_normal(X1.shape).astype(np.float32)   # 随机噪声 (起点)
        t = rng.random((len(X1), 1)).astype(np.float32)         # 随机时间 [0,1]
        xt = (1 - t) * x0 + t * X1                              # 直线插值
        target_v = X1 - x0                                      # 直线速度 (恒定)
        pred_v = model(torch.tensor(xt), torch.tensor(t))
        loss = lossf(pred_v, torch.tensor(target_v))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def sample(model, n: int = 512, dim: int = 2, steps: int = 10, seed: int = 0,
           record_traj: bool = False):
    """ODE Euler 积分采样: 从噪声出发, 沿速度场走 `steps` 步到数据.
    steps 可以很小 (flow matching 的卖点: 少步也能生成)。"""
    import torch
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    dt = 1.0 / steps
    traj = [x.copy()]
    with torch.no_grad():
        for i in range(steps):
            t = np.full((n, 1), i * dt, dtype=np.float32)
            v = model(torch.tensor(x), torch.tensor(t)).numpy()
            x = x + v * dt                          # Euler step: 沿速度走一步
            if record_traj:
                traj.append(x.copy())
    return (x, traj) if record_traj else x


def reflow(model, x1: np.ndarray, epochs: int = 500, lr: float = 2e-3, seed: int = 0):
    """rectified flow 的「重流 (reflow)」(M13.2-L2): 用模型自己生成的 (噪声→数据) 配对重训,
    让路径更直 (少步采样质量更好). 返回 (新模型, loss 历史)。

    思想: 第一轮模型把噪声 x0 流到数据 x1'。这些 (x0, x1') 是「模型连好的」配对, 不再随机
    交叉。用它们重训, 学到的路径更直。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    # 1. 用当前模型生成「配对」: 固定 x0, 沿速度场流到 x1'
    x0 = rng.standard_normal(x1.shape).astype(np.float32)
    x1_paired = _flow_from(model, x0, steps=20)
    # 2. 用这些不交叉的配对重训一个新模型
    new_model = build_velocity_field(dim=x1.shape[1])
    torch.manual_seed(seed)
    opt = torch.optim.Adam(new_model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        idx = rng.integers(0, len(x0), size=len(x0))
        a, b = x0[idx], x1_paired[idx]
        t = rng.random((len(idx), 1)).astype(np.float32)
        xt = ((1 - t) * a + t * b).astype(np.float32)
        target_v = (b - a).astype(np.float32)
        pred = new_model(torch.tensor(xt), torch.tensor(t))
        loss = lossf(pred, torch.tensor(target_v))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return new_model, losses


def _flow_from(model, x0: np.ndarray, steps: int = 20) -> np.ndarray:
    """从给定 x0 沿速度场流到数据 (固定起点, 用于 reflow 配对)。"""
    import torch
    x = x0.copy(); dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t = np.full((len(x), 1), i * dt, dtype=np.float32)
            x = x + model(torch.tensor(x), torch.tensor(t)).numpy() * dt
    return x


def quality_vs_steps(model, x1: np.ndarray, step_list=(2, 4, 8, 16, 32),
                     seed: int = 0) -> list[dict]:
    """不同采样步数下生成质量 (用生成分布与目标的统计差衡量). flow matching 少步也行。"""
    out = []
    target_std = x1.std(0)
    for s in step_list:
        gen = sample(model, n=400, dim=x1.shape[1], steps=s, seed=seed)
        std_err = float(np.abs(gen.std(0) - target_std).mean())
        out.append({"steps": s, "std_err": round(std_err, 4)})
    return out


if __name__ == "__main__":
    x1 = make_two_moons(n=512, seed=1)
    model = build_velocity_field()
    if model is not None:
        losses = train_flow_matching(model, x1, epochs=500)
        print(f"flow matching 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}")
        print("\n采样步数 → 生成质量 (std 误差, 越小越好):")
        for r in quality_vs_steps(model, x1):
            print(f"  {r['steps']:>3} 步: std_err {r['std_err']}")
        print("→ flow matching 用很少步 (如 4-8) 就能生成 (路径近直, 比 DDPM 省步)。")
