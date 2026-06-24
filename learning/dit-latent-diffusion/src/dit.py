"""
dit.py — Diffusion Transformer (DiT) + classifier-free guidance (CFG), 扩散的「骨架」.

为什么需要它 (M13.3): M13.1/13.2 的去噪/速度网络是个 MLP。真实扩散模型 (SD3/Sora/DiT) 用
**transformer** 当去噪网络 —— 这是你的本行! 而且要能**条件生成** (按类别/文本生成指定内容),
靠 **classifier-free guidance (CFG)** 控制「跟条件跟得多紧」。

本文件在一个多类 2D 数据 (4 个高斯团 = 4 类) 上演示:
  - DiT 式去噪器: 把点当 token, 用 transformer + 时间/类别条件去噪
  - 条件生成: 指定类别, 生成对应那一团
  - CFG: 用一个 guidance scale 旋钮控制「生成多贴近指定类别」(强度消融)

纯 torch (tiny CPU), 确定性。机制和真实 DiT 一致, 只是规模小。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NUM_CLASSES = 4
NULL_CLASS = NUM_CLASSES   # 用于 CFG 的「无条件」标签


def make_class_blobs(n_per: int = 200, seed: int = 0):
    """4 个高斯团 (四角), 每团一个类别. 返回 (x, y) = (点, 类别)。条件扩散要按类生成。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=np.float32)
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        xs.append(centers[c] + 0.4 * rng.standard_normal((n_per, 2)))
        ys.append(np.full(n_per, c))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    x = (x / 2.5).astype(np.float32)   # 标准化
    return x, y


def make_beta_schedule(T: int = 50):
    betas = np.linspace(1e-4, 0.2, T).astype(np.float32)
    alpha_bars = np.cumprod(1 - betas)
    return betas, (1 - betas), alpha_bars


def build_dit(dim: int = 2, d_model: int = 48, n_layers: int = 2, seed: int = 0):
    """DiT 式去噪器: 点→token, transformer 处理, 条件 = 时间 + 类别 embedding. 无 torch 返回 None。"""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[dit] 无 torch ({exc!r})")
        return None
    torch.manual_seed(seed)

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(dim, d_model)
            self.t_embed = nn.Linear(1, d_model)               # 时间条件
            self.c_embed = nn.Embedding(NUM_CLASSES + 1, d_model)  # 类别条件 (+1 = null, CFG 用)
            self.blocks = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 4, d_model * 2, batch_first=True), n_layers)
            self.out = nn.Linear(d_model, dim)

        def forward(self, x, t, c):  # x:(B,dim), t:(B,1), c:(B,) 类别
            import torch
            # 把点当一个 token; 把时间和类别 embedding 当额外 token (条件 token, DiT 思想)
            h = self.in_proj(x).unsqueeze(1)                   # (B,1,d)
            cond = (self.t_embed(t) + self.c_embed(c)).unsqueeze(1)  # (B,1,d)
            seq = torch.cat([cond, h], dim=1)                  # [条件 token | 数据 token]
            seq = self.blocks(seq)
            return self.out(seq[:, 1])                         # 取数据 token 输出预测噪声

    return DiT()


def train_dit(model, x: np.ndarray, y: np.ndarray, T: int = 50, epochs: int = 600,
              lr: float = 2e-3, p_uncond: float = 0.15, seed: int = 0):
    """训练 DiT (预测噪声) + CFG 准备: 以 p_uncond 概率把类别换成 null, 让模型同时学
    「有条件」和「无条件」去噪 (CFG 需要两者)。返回 (loss 历史, schedule)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    betas, alphas, abars = make_beta_schedule(T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        t = rng.integers(0, T, size=len(x))
        ab = abars[t][:, None]
        eps = rng.standard_normal(x.shape).astype(np.float32)
        xt = (np.sqrt(ab) * x + np.sqrt(1 - ab) * eps).astype(np.float32)
        # CFG: 随机丢弃类别条件 (换成 null), 让模型学无条件去噪
        c = y.copy()
        drop = rng.random(len(x)) < p_uncond
        c[drop] = NULL_CLASS
        pred = model(torch.tensor(xt), torch.tensor((t[:, None] / T).astype(np.float32)),
                     torch.tensor(c))
        loss = lossf(pred, torch.tensor(eps))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, (betas, alphas, abars)


def sample(model, schedule, cls: int, n: int = 200, T_steps: int = None,
           guidance: float = 1.0, seed: int = 0):
    """条件采样: 生成类别 cls 的样本. guidance = CFG 强度 (1=普通, >1 更贴近类别).

    CFG 公式: eps = eps_uncond + guidance * (eps_cond - eps_uncond)
    guidance 越大, 越往「有条件」方向放大 → 生成越贴近指定类别 (但太大会过饱和)。
    """
    import torch
    betas, alphas, abars = schedule
    T = len(betas)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    c_cond = torch.full((n,), cls, dtype=torch.long)
    c_null = torch.full((n,), NULL_CLASS, dtype=torch.long)
    with torch.no_grad():
        for t in reversed(range(T)):
            tn = torch.full((n, 1), t / T)
            eps_c = model(torch.tensor(x), tn, c_cond).numpy()
            eps_u = model(torch.tensor(x), tn, c_null).numpy()
            eps = eps_u + guidance * (eps_c - eps_u)           # CFG
            ab = abars[t]; a = alphas[t]; b = betas[t]
            mean = (1 / np.sqrt(a)) * (x - (b / np.sqrt(1 - ab)) * eps)
            if t > 0:
                x = mean + np.sqrt(b) * rng.standard_normal((n, 2)).astype(np.float32)
            else:
                x = mean
    return x


def class_accuracy(samples: np.ndarray, target_cls: int) -> float:
    """生成样本落在目标类中心附近的比例 (衡量条件生成是否对)。"""
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=np.float32) / 2.5
    d = ((samples[:, None] - centers[None]) ** 2).sum(-1)
    pred = d.argmin(1)
    return float((pred == target_cls).mean())


if __name__ == "__main__":
    x, y = make_class_blobs(n_per=200, seed=1)
    model = build_dit()
    if model is not None:
        losses, sched = train_dit(model, x, y, epochs=600)
        print(f"DiT 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}")
        print("\n条件生成 + CFG 强度 → 类别准确率 (生成类别 0):")
        for g in [0.0, 1.0, 2.0, 4.0]:
            s = sample(model, sched, cls=0, n=200, guidance=g, seed=2)
            print(f"  guidance={g}: 类别准确率 {class_accuracy(s, 0):.2f}")
        print("→ guidance 越大, 生成越贴近指定类别 (CFG, M13.3-L3)。")
