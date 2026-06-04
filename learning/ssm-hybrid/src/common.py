"""SSM 共用工具 — discretization helpers."""
from __future__ import annotations

import torch


def discretize_zoh(A: torch.Tensor, B: torch.Tensor,
                   delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-Order Hold discretization.

    A_bar = exp(delta * A)
    B_bar = (exp(delta * A) - I) / A * B
    """
    dA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(dA)
    B_bar = (A_bar - 1) / A * B
    return A_bar, B_bar


def naive_scan(A: torch.Tensor, B: torch.Tensor,
               u: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Naive sequential scan (slow but correct).

    A: (t, d_state)
    B: (t, d_state)
    u: (t,)
    C: (t, d_state)
    Returns y: (t,)
    """
    t = u.shape[0]
    d_state = A.shape[-1]
    x = torch.zeros(d_state, device=u.device, dtype=u.dtype)
    y = torch.zeros(t, device=u.device, dtype=u.dtype)
    for i in range(t):
        x = A[i] * x + B[i] * u[i]
        y[i] = (C[i] * x).sum()
    return y
