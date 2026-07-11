"""真训练循环 + warmup/cosine 学习率，从零手写。

面试高频度 ★★★★★（"写个训练循环"）。得分点：zero_grad→backward→clip→step 四步不漏；
warmup 防早期 Adam 方差估计不稳，cosine 尾段退火。LR schedule 抽成纯函数单独测边界。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_warmup_lr(step: int, warmup: int, total: int,
                     base_lr: float, min_lr: float = 0.0) -> float:
    """线性 warmup 到 base_lr，然后 cosine 退火到 min_lr。"""
    if step < warmup:                                   # 线性升
        return base_lr * (step + 1) / warmup
    if step >= total:                                   # 训练结束后钳住
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)  # 0..1
    cos = 0.5 * (1 + math.cos(math.pi * progress))       # 1..0
    return min_lr + (base_lr - min_lr) * cos


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def train(model, X, Y, steps=300, base_lr=1e-2, warmup=30, clip=1.0):
    """标准四步循环 + 手动调度器。返回 (loss_history)。"""
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    loss_fn = nn.MSELoss()
    history = []
    for step in range(steps):
        lr = cosine_warmup_lr(step, warmup, steps, base_lr, min_lr=base_lr * 0.05)
        for g in opt.param_groups:                      # 手动写 LR
            g["lr"] = lr
        opt.zero_grad()                                 # ① 清零
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()                                 # ② 反传
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # ③ 裁剪
        opt.step()                                      # ④ 更新
        history.append(loss.item())
    return history


def _self_test() -> None:
    # LR schedule 边界
    assert abs(cosine_warmup_lr(0, 10, 100, 1.0) - 0.1) < 1e-9      # 第0步 = base/warmup
    assert abs(cosine_warmup_lr(9, 10, 100, 1.0) - 1.0) < 1e-9      # warmup 末 = base
    assert cosine_warmup_lr(10, 10, 100, 1.0) <= 1.0               # 峰后开始退火
    assert abs(cosine_warmup_lr(100, 10, 100, 1.0, min_lr=0.05) - 0.05) < 1e-9  # 结束=min
    # warmup 段单调升
    warm = [cosine_warmup_lr(s, 10, 100, 1.0) for s in range(10)]
    assert all(warm[i] < warm[i + 1] for i in range(9))

    # 真训练：拟合非线性映射，loss 应大幅下降
    torch.manual_seed(0)
    X = torch.randn(128, 8)
    W = torch.randn(8, 4)
    Y = torch.tanh(X @ W) + 0.1 * torch.randn(128, 4)   # 非线性目标 → 需要 MLP
    model = TinyMLP(8, 32, 4)
    hist = train(model, X, Y, steps=300)
    assert hist[-1] < 0.5 * hist[0], f"loss 未下降: {hist[0]:.3f}->{hist[-1]:.3f}"
    assert math.isfinite(hist[-1])
    print(f"[PASS] training_loop: LR边界 + 单调warmup + 真拟合 "
          f"({hist[0]:.3f}->{hist[-1]:.3f})")


if __name__ == "__main__":
    _self_test()
