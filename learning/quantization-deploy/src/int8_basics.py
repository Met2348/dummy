"""int8 quantization basics — per-tensor / per-channel / per-group."""
from __future__ import annotations

import torch


def quantize_per_tensor(x: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, float]:
    """Symmetric int8 quantization with a single scale."""
    qmax = (1 << (n_bits - 1)) - 1
    scale = x.abs().max().item() / max(qmax, 1)
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_per_channel(x: torch.Tensor, axis: int = 0, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel along `axis`."""
    qmax = (1 << (n_bits - 1)) - 1
    # compute scale along the OTHER axes
    keep_dims = list(range(x.ndim))
    keep_dims.remove(axis)
    scale = x.abs().amax(dim=keep_dims, keepdim=True) / max(qmax, 1)
    scale = scale.clamp(min=1e-9)
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_per_group(x: torch.Tensor, group_size: int = 128, n_bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric quantization along last dim."""
    qmax = (1 << (n_bits - 1)) - 1
    *prefix, K = x.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    g = K // group_size
    x_grouped = x.reshape(*prefix, g, group_size)
    scale = x_grouped.abs().amax(dim=-1, keepdim=True) / max(qmax, 1)
    scale = scale.clamp(min=1e-9)
    q = (x_grouped / scale).round().clamp(-qmax, qmax)
    return q.reshape(*prefix, K).to(torch.int8), scale.squeeze(-1)


def dequantize_per_group(q: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    *prefix, K = q.shape
    g = K // group_size
    q_grouped = q.reshape(*prefix, g, group_size).to(torch.float32)
    return (q_grouped * scale.unsqueeze(-1)).reshape(*prefix, K)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())
