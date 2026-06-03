"""共用工具 — numerical safe softmax / causal mask / helpers."""
from __future__ import annotations

import math

import torch


def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 softmax（减 max）."""
    x_max = x.amax(dim=dim, keepdim=True)
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


def causal_mask(t: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """下三角 1, 上三角 -inf 的 attention mask, shape (t, t)."""
    return torch.full((t, t), float("-inf"), device=device, dtype=dtype).triu_(1)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_weights_(module: torch.nn.Module, std: float = 0.02) -> None:
    """GPT-2-style init: Linear/Embedding → normal(0, std)."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
