"""Pretraining recipe 公共工具."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cosine_lr(step: int, max_step: int, base_lr: float,
              min_lr: float = 1e-5, warmup: int = 1000) -> float:
    """warmup + cosine decay."""
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_step - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def wsd_lr(step: int, max_step: int, base_lr: float,
            warmup_pct: float = 0.05, decay_pct: float = 0.2) -> float:
    """Warmup-Stable-Decay (Phi/MiniCPM)."""
    warmup = int(max_step * warmup_pct)
    decay_start = int(max_step * (1 - decay_pct))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    progress = (step - decay_start) / (max_step - decay_start)
    return base_lr * (1 - progress)


def lr_inverse_sqrt(step: int, base_lr: float, warmup: int = 4000) -> float:
    if step < warmup:
        return base_lr * step / warmup
    return base_lr * (warmup / step) ** 0.5


def chinchilla_optimal_tokens(n_params: int) -> int:
    """1:20 ratio."""
    return n_params * 20


def estimate_flops(n_params: int, n_token: int) -> float:
    """6ND 估算训练 FLOPs."""
    return 6 * n_params * n_token


if __name__ == "__main__":
    print("LR schedules @ step 500/5000:")
    for s in [500, 1000, 2500, 4500, 5000]:
        print(f"  step {s}: cosine={cosine_lr(s, 5000, 1e-3):.4e} "
              f"wsd={wsd_lr(s, 5000, 1e-3):.4e}")
