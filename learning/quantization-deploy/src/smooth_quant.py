"""SmoothQuant — move activation outliers into weights."""
from __future__ import annotations

import torch


def find_smooth_scale(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-channel scale `s` such that X/s and diag(s)·W are both 'flatter'.

    Convention: X is `[N, K]`, W is `[K, M]`, output Y = X · W is `[N, M]`.
    `s` has shape `[K]` — the shared inner dimension.

    `s_k = max(|X_:k|)^α / max(|W_k:|)^(1-α)`
    """
    x_amax = X.abs().amax(dim=0).clamp(min=1e-9)           # [K]
    w_amax = W.abs().amax(dim=1).clamp(min=1e-9)           # [K]
    s = x_amax.pow(alpha) / w_amax.pow(1.0 - alpha)
    return s.clamp(min=1e-3)


def apply_smoothing(X: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`Y = X·W = (X / s) · (diag(s)·W)`.

    - `X_s = X / s` divides each column k of X by s_k.
    - `W_s = s[:, None] * W` multiplies each row k of W by s_k.
    """
    return X / s, s.unsqueeze(-1) * W


def smooth_quant_matmul_mock(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Run X · W after smoothing — purely a teaching demo."""
    s = find_smooth_scale(X, W, alpha)
    X_s, W_s = apply_smoothing(X, W, s)
    return X_s @ W_s
