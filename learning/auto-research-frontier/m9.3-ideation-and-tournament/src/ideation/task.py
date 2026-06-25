"""一个真·可跑的 toy 任务：numpy 逻辑回归 + 梯度下降。

每个"研究 idea"在这里是一组**训练配置**；它的"可行性 feasibility"= 真跑出来的准确率，
不是硬编码的数。于是"听着新颖的 idea 真做出来反而更差"这件事，长在真实测量上
（这是 9.5 ideation-execution gap 的排序版）。
"""
from __future__ import annotations

import numpy as np

# 默认配置：一个能跑、但远未到顶的 baseline（步数有限，留出"多训练"的提升空间）
DEFAULT = dict(lr=0.04, steps=250, momentum=0.0, init_scale=0.1,
               lr_grows=False, data_seed=0, init_seed=0)


def make_data(n: int = 400, seed: int = 0):
    """**病态条件**数据：两个特征尺度差 20 倍 → 单一全局 lr 很难同时照顾到两维，
    于是"调 lr/加动量/多训"这些优化选择真的会拉开差距，坏选择会发散或欠拟合。"""
    rng = np.random.default_rng(seed)
    X = np.stack([rng.normal(0, 1.0, n), rng.normal(0, 20.0, n)], axis=1)
    w_true, b_true = np.array([2.0, -0.13]), 0.2
    logits = X @ w_true + b_true + rng.normal(scale=0.8, size=n)
    y = (logits > 0).astype(float)
    return X, y


def train_logreg(config: dict) -> float:
    """真跑 GD，返回训练集准确率（确定性，可复现）。"""
    cfg = {**DEFAULT, **config}
    X, y = make_data(seed=cfg["data_seed"])
    rng = np.random.default_rng(cfg["init_seed"])
    w = rng.normal(size=2) * cfg["init_scale"]
    b = 0.0
    vw, vb = np.zeros(2), 0.0
    n = len(y)
    for t in range(cfg["steps"]):
        z = np.clip(X @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        gw = X.T @ (p - y) / n
        gb = float((p - y).mean())
        # "fancy 新调度"：让 lr 随训练放大——听着高级，其实容易发散
        lr_t = cfg["lr"] * (1.0 + 5.0 * t / cfg["steps"]) if cfg["lr_grows"] else cfg["lr"]
        vw = cfg["momentum"] * vw + gw
        vb = cfg["momentum"] * vb + gb
        w = w - lr_t * vw
        b = b - lr_t * vb
        if not np.all(np.isfinite(w)):   # 发散：直接判一个差分
            return 0.5
    z = np.clip(X @ w + b, -30, 30)
    acc = float((((1.0 / (1.0 + np.exp(-z))) > 0.5) == (y > 0.5)).mean())
    return acc
