"""Minimal GPTQ — column-by-column quantization with Hessian correction."""
from __future__ import annotations

import torch


def gptq_quantize(
    W: torch.Tensor,      # [out_features, in_features]
    H: torch.Tensor,      # [in_features, in_features] hessian (X^T X)
    n_bits: int = 4,
    damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-layer GPTQ.

    Returns (quantized W as floats reconstructed, per-channel scales).
    """
    out_dim, in_dim = W.shape
    qmax = (1 << (n_bits - 1)) - 1
    Wq = W.clone().float()

    # Cholesky-friendly Hessian inverse
    Hreg = H + damp * torch.eye(in_dim, device=H.device) * H.diag().mean()
    L = torch.linalg.cholesky(Hreg)
    H_inv = torch.cholesky_inverse(L)

    scales: list[torch.Tensor] = []
    for j in range(in_dim):
        # quantize column j
        col = Wq[:, j]
        s = col.abs().max() / max(qmax, 1)
        s = s.clamp(min=1e-9)
        q_int = (col / s).round().clamp(-qmax, qmax)
        col_q = q_int * s
        err = (col - col_q) / H_inv[j, j]
        # distribute error to remaining columns
        if j + 1 < in_dim:
            Wq[:, j + 1 :] -= torch.outer(err, H_inv[j, j + 1 :])
        Wq[:, j] = col_q
        scales.append(s)

    return Wq, torch.stack(scales)


def calibrate_hessian(X: torch.Tensor) -> torch.Tensor:
    """Given calibration activations X [N, in_features], return H = X^T X / N."""
    return X.float().t() @ X.float() / max(X.shape[0], 1)
