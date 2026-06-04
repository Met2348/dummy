"""Minimal AWQ — search per-channel scale that minimises quant error."""
from __future__ import annotations

import torch
from int8_basics import quantize_per_channel, dequantize_per_channel


def search_scales(
    W: torch.Tensor,           # [out_features, in_features]
    X: torch.Tensor,           # [N, in_features] calibration
    n_bits: int = 4,
    grid: int = 20,
) -> torch.Tensor:
    """Grid-search per-channel scale s ∈ (0, 1]; minimise ||(W s) X(1/s) - W X||²."""
    out_dim, in_dim = W.shape
    x_mean = X.abs().mean(dim=0).clamp(min=1e-9)   # [in_dim]
    best_s = torch.ones(in_dim, device=W.device)
    best_err = float("inf")
    ref = W @ X.t()           # [out, N] reference matmul

    for ratio in [1.0 - i / grid for i in range(grid)]:
        s = x_mean.pow(ratio).clamp(min=1e-3)      # [in_dim]
        W_s = W * s                                 # [out, in]
        Wq_s, scale = quantize_per_channel(W_s, axis=0, n_bits=n_bits)
        W_dq = dequantize_per_channel(Wq_s, scale) / s   # undo scaling
        approx = W_dq @ X.t()
        err = float(((approx - ref) ** 2).mean().item())
        if err < best_err:
            best_err = err
            best_s = s.clone()
    return best_s


def awq_quantize(
    W: torch.Tensor,
    X: torch.Tensor,
    n_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = search_scales(W, X, n_bits=n_bits)
    W_s = W * s
    Wq, scale = quantize_per_channel(W_s, axis=0, n_bits=n_bits)
    return Wq, s
