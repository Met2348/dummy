"""阶段 3：真实实验执行（不是 mock 分数）。

在一个 nonlinear 的合成二分类任务（make_moons）上真训练一个小 MLP，返回真实指标。
关键设计：
- **任务种子 data_seed 与模型种子 seed 分离**：对照 baseline vs treatment 时，
  数据集（任务）固定不变，只变模型/训练——这是公平实验的基本功。
- **确定性**：固定 seed → 同 config 同结果（test 里用 CPU 锁死复现）。
- 跨多个 seed 重复 → 给出 mean±std，让后续 analysis 能做"带误差棒的比较"
  （而不是单跑一次就下结论——这正是 AI Scientist 常踩的坑）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons


@dataclass
class ExperimentConfig:
    # 任务（对照时保持不变）
    n_samples: int = 600
    noise: float = 0.2
    data_seed: int = 0
    # 模型 / 训练（idea 在这里做改动）
    depth: int = 1            # 隐藏层数；0 = 线性模型
    hidden_dim: int = 16
    activation: str = "relu"  # relu | tanh
    lr: float = 0.05
    epochs: int = 200
    seed: int = 0             # 模型初始化/训练种子（重复实验时变这个）
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: ExperimentConfig):
    """同一个 data_seed 永远给同一个任务（与模型种子解耦）。"""
    X, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise, random_state=cfg.data_seed)
    rng = np.random.RandomState(cfg.data_seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_test = len(X) // 3
    dev = cfg.device
    Xtr = torch.tensor(X[n_test:], dtype=torch.float32, device=dev)
    ytr = torch.tensor(y[n_test:], dtype=torch.long, device=dev)
    Xte = torch.tensor(X[:n_test], dtype=torch.float32, device=dev)
    yte = torch.tensor(y[:n_test], dtype=torch.long, device=dev)
    return Xtr, ytr, Xte, yte


class MLP(nn.Module):
    def __init__(self, depth: int, hidden_dim: int, activation: str):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), act()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))   # 2-class logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _accuracy(model: nn.Module, X, y) -> float:
    return (model(X).argmax(1) == y).float().mean().item()


def run_experiment(cfg: ExperimentConfig) -> dict:
    """真训练一次，返回真实指标（确定性：同 cfg → 同结果）。"""
    dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg.device = dev
    Xtr, ytr, Xte, yte = build_dataset(cfg)
    set_seed(cfg.seed)
    model = MLP(cfg.depth, cfg.hidden_dim, cfg.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    final_loss = float("nan")
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)   # full-batch GD（小数据，快且确定）
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return {
        "test_acc": round(_accuracy(model, Xte, yte), 4),
        "train_acc": round(_accuracy(model, Xtr, ytr), 4),
        "final_loss": round(final_loss, 4),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def run_repeated(cfg: ExperimentConfig, seeds: List[int]) -> dict:
    """跨多个模型种子重复（任务不变），给出 mean±std——后续比较的基础。"""
    accs = []
    last = {}
    for s in seeds:
        c = ExperimentConfig(**{**cfg.__dict__, "seed": s})
        last = run_experiment(c)
        accs.append(last["test_acc"])
    arr = np.array(accs, dtype=float)
    return {
        "test_acc_mean": round(float(arr.mean()), 4),
        "test_acc_std": round(float(arr.std()), 4),
        "test_accs": accs,
        "n_params": last.get("n_params"),
        "seeds": list(seeds),
    }
