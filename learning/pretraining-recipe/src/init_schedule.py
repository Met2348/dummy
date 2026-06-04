"""模型初始化 + 学习率 schedule 实现."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def init_gpt2_style(module: nn.Module, n_layer: int = 12) -> None:
    """GPT-2 style: N(0, 0.02), residual proj scaled."""
    for name, p in module.named_parameters():
        if "weight" in name and p.dim() >= 2:
            std = 0.02
            if "c_proj" in name or "out_proj" in name:
                std /= math.sqrt(2 * n_layer)
            nn.init.normal_(p, mean=0, std=std)
        elif "bias" in name:
            nn.init.zeros_(p)


def init_mup_style(module: nn.Module, base_width: int = 256) -> None:
    """μP style: σ ∝ 1/√fan_in, output proj 额外缩放."""
    for name, p in module.named_parameters():
        if "weight" in name and p.dim() >= 2:
            fan_in = p.shape[-1]
            std = math.sqrt(1.0 / fan_in)
            if "lm_head" in name:
                std *= math.sqrt(base_width / fan_in)
            nn.init.normal_(p, std=std)


def cosine_with_warmup(step, max_step, base_lr, min_lr=1e-5, warmup=2000):
    if step < warmup:
        return base_lr * step / warmup
    t = (step - warmup) / max(1, max_step - warmup)
    t = min(1.0, t)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def wsd(step, max_step, base_lr, warmup_pct=0.05, decay_pct=0.2):
    warmup = int(max_step * warmup_pct)
    decay_start = int(max_step * (1 - decay_pct))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    t = (step - decay_start) / (max_step - decay_start)
    return base_lr * (1 - t)


def inverse_sqrt(step, base_lr, warmup=4000):
    if step < warmup:
        return base_lr * step / warmup
    return base_lr * math.sqrt(warmup / step)


def mup_lr(width_now: int, width_base: int, lr_base: float) -> float:
    """μP: scale lr with width."""
    return lr_base * (width_base / width_now)


if __name__ == "__main__":
    print("=== Schedule curves ===")
    for s in [0, 100, 1000, 2500, 4500, 5000]:
        print(f"  step {s}: "
              f"cosine={cosine_with_warmup(s, 5000, 6e-4):.3e} "
              f"wsd={wsd(s, 5000, 6e-4):.3e} "
              f"isqrt={inverse_sqrt(s, 6e-4):.3e}")

    print("\n=== μP lr scaling ===")
    for w in [256, 512, 1024, 2048, 4096]:
        print(f"  width={w}: lr={mup_lr(w, 256, 1e-3):.4e}")
