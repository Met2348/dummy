"""
diffusion.py — 最小 DDPM (去噪扩散), 在 2D 玩具分布上看见「扩散在干什么」.

为什么从这里开始 (M13 地基, 也被 M11 diffusion policy 复用): 扩散是和自回归并列的另一根生成
支柱。核心思想反直觉但简单: **逐步给数据加噪声直到变成纯噪声 (前向); 学一个网络把这个过程
反过来, 从纯噪声逐步去噪还原出数据 (反向)。生成 = 从噪声去噪。**

本文件在 2D 玩具分布 (双月/螺旋) 上实现最小 DDPM:
  - 前向加噪 q(x_t|x_0): 一步到位的闭式
  - 去噪网络 ε_θ: 一个 tiny MLP, 预测「加了什么噪声」
  - 训练: 让网络预测噪声
  - 采样: 从纯噪声反向逐步去噪, 还原数据分布
  - 可视化去噪轨迹 (看见点云从噪声收敛成双月)

纯 torch (tiny, CPU 秒级), 确定性 (固定 seed), 离线。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_two_moons(n: int = 512, noise: float = 0.08, seed: int = 0) -> np.ndarray:
    """2D 双月分布 (经典玩具数据). 返回 (n, 2)。扩散要学会生成这个形状。"""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = np.pi * rng.random(n1)
    moon1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    t2 = np.pi * rng.random(n - n1)
    moon2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], 1)
    x = np.concatenate([moon1, moon2], 0)
    x = x + noise * rng.standard_normal(x.shape)
    return ((x - x.mean(0)) / x.std(0)).astype(np.float32)   # 标准化


def make_beta_schedule(T: int = 50, beta_start: float = 1e-4, beta_end: float = 0.2):
    """线性 beta 调度. 返回 (betas, alphas, alpha_bars). 这些控制每步加多少噪声。"""
    betas = np.linspace(beta_start, beta_end, T).astype(np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def forward_noise(x0: np.ndarray, t: int, alpha_bars: np.ndarray, rng) -> tuple:
    """前向加噪 q(x_t|x_0) 闭式: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε. 返回 (x_t, ε)。
    一步到位 (不用真的迭代 t 步), 这是 DDPM 训练高效的关键。"""
    ab = alpha_bars[t]
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    return xt, eps


def build_denoiser(dim: int = 2, hidden: int = 64, seed: int = 0):
    """去噪网络 ε_θ(x_t, t): 一个 tiny MLP, 输入带噪点 + 时间步, 预测噪声. 无 torch 返回 None。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[diffusion] 无 torch ({exc!r})")
        return None
    torch.manual_seed(seed)

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, dim))

        def forward(self, xt, t_norm):  # xt: (B,dim), t_norm: (B,1) 归一化时间
            import torch
            return self.net(torch.cat([xt, t_norm], -1))

    return Denoiser()


def train_diffusion(model, x0: np.ndarray, T: int = 50, epochs: int = 400,
                    lr: float = 2e-3, seed: int = 0):
    """训练去噪网络: 随机取 t, 加噪, 让网络预测加的噪声. 返回 (loss 历史, schedule)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    betas, alphas, alpha_bars = make_beta_schedule(T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    X = torch.tensor(x0)
    losses = []
    for _ in range(epochs):
        t = rng.integers(0, T, size=len(x0))
        ab = alpha_bars[t][:, None]
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        xt = (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32)
        t_norm = (t[:, None] / T).astype(np.float32)
        pred = model(torch.tensor(xt), torch.tensor(t_norm))
        loss = lossf(pred, torch.tensor(eps))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, (betas, alphas, alpha_bars)


def sample(model, schedule, n: int = 512, dim: int = 2, seed: int = 0,
           record_traj: bool = False):
    """反向采样: 从纯噪声 x_T 逐步去噪到 x_0 (DDPM ancestral sampling). 返回生成样本。
    record_traj=True 时返回 (samples, 轨迹列表) 用于可视化去噪过程。"""
    import torch
    betas, alphas, alpha_bars = schedule
    T = len(betas)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)   # 从纯噪声开始
    traj = [x.copy()]
    with torch.no_grad():
        for t in reversed(range(T)):
            t_norm = np.full((n, 1), t / T, dtype=np.float32)
            eps_pred = model(torch.tensor(x), torch.tensor(t_norm)).numpy()
            ab = alpha_bars[t]; a = alphas[t]; b = betas[t]
            # DDPM 反向均值: 用预测噪声去掉一步噪声
            mean = (1 / np.sqrt(a)) * (x - (b / np.sqrt(1 - ab)) * eps_pred)
            if t > 0:
                noise = rng.standard_normal((n, dim)).astype(np.float32)
                x = mean + np.sqrt(b) * noise
            else:
                x = mean
            if record_traj:
                traj.append(x.copy())
    return (x, traj) if record_traj else x


if __name__ == "__main__":
    x0 = make_two_moons(n=512, seed=1)
    print(f"目标分布: 双月, {x0.shape}")
    model = build_denoiser()
    if model is not None:
        losses, sched = train_diffusion(model, x0, T=50, epochs=400)
        print(f"训练: loss {losses[0]:.3f} → {losses[-1]:.3f}")
        gen = sample(model, sched, n=512, seed=2)
        # 简单检验: 生成样本的统计接近目标
        print(f"目标 std: {x0.std(0).round(2)}, 生成 std: {gen.std(0).round(2)}")
        print("→ 从纯噪声去噪, 还原出双月分布。生成 = 去噪 (M13-L3)。")
