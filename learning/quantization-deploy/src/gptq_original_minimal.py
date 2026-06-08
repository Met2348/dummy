"""Paper-shaped toy examples for GPTQ.

The real GPTQ implementation is a large PyTorch/CUDA engineering project.
This file keeps only the mechanism that matters for reading the paper:

- build a layer reconstruction objective from calibration activations;
- approximate the layer Hessian with X^T X;
- quantize a weight matrix column by column;
- compensate each column's quantization error on later columns through H^-1.

Run:
    .\\.venv\\Scripts\\python.exe learning\\quantization-deploy\\src\\gptq_original_minimal.py
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SymmetricRowGrid:
    """A simple per-output-row symmetric grid.

    The paper uses uniform per-row asymmetric min-max grids in the main
    experiments. This toy grid is symmetric so that the mechanism is easy to
    inspect, but it keeps the important property that each output row has its
    own scale.
    """

    scale: torch.Tensor
    qmax: int


def make_symmetric_row_grid(W: torch.Tensor, n_bits: int = 4) -> SymmetricRowGrid:
    """Return one scale per output row for signed integer quantization."""
    if n_bits < 2:
        raise ValueError("n_bits must be at least 2 for signed quantization")
    qmax = (1 << (n_bits - 1)) - 1
    scale = W.float().abs().amax(dim=1).clamp(min=1e-9) / qmax
    return SymmetricRowGrid(scale=scale, qmax=qmax)


def quantize_column(col: torch.Tensor, grid: SymmetricRowGrid) -> torch.Tensor:
    """Round one column to the nearest value on each row's grid."""
    q = (col.float() / grid.scale).round().clamp(-grid.qmax, grid.qmax)
    return q * grid.scale


def naive_round(W: torch.Tensor, n_bits: int = 4) -> torch.Tensor:
    """Quantize all weights independently with no Hessian compensation."""
    grid = make_symmetric_row_grid(W, n_bits=n_bits)
    q = (W.float() / grid.scale[:, None]).round().clamp(-grid.qmax, grid.qmax)
    return q * grid.scale[:, None]


def calibration_hessian(X: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Return a damped Hessian proxy for layer inputs X [samples, in_dim]."""
    X = X.float()
    in_dim = X.shape[1]
    H = 2.0 * (X.t() @ X) / max(X.shape[0], 1)
    diagonal_mean = H.diag().mean().clamp(min=1e-9)
    return H + damp * diagonal_mean * torch.eye(in_dim, device=X.device)


def gptq_columnwise(
    W: torch.Tensor,
    X: torch.Tensor,
    n_bits: int = 4,
    damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize W with the core GPTQ column-by-column compensation rule.

    Shape convention:
        W: [out_features, in_features]
        X: [samples, in_features]

    Returns:
        Q: quantized-dequantized weight matrix
        E: per-column compensation errors before they are pushed forward
    """
    W_work = W.float().clone()
    out_dim, in_dim = W_work.shape
    grid = make_symmetric_row_grid(W_work, n_bits=n_bits)
    H = calibration_hessian(X, damp=damp)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    Q = torch.zeros_like(W_work)
    E = torch.zeros_like(W_work)
    for j in range(in_dim):
        q_col = quantize_column(W_work[:, j], grid)
        err = (W_work[:, j] - q_col) / H_inv[j, j]

        Q[:, j] = q_col
        E[:, j] = err
        if j + 1 < in_dim:
            W_work[:, j + 1 :] -= torch.outer(err, H_inv[j, j + 1 :])

    return Q, E


def reconstruction_mse(W: torch.Tensor, Q: torch.Tensor, X: torch.Tensor) -> float:
    """Squared layer-output error for y = W x over calibration examples."""
    full = W.float() @ X.float().t()
    quantized = Q.float() @ X.float().t()
    return torch.mean((full - quantized) ** 2).item()


def toy_comparison(seed: int = 0) -> dict[str, float]:
    """Return a reproducible RTN-vs-GPTQ comparison on one toy layer."""
    torch.manual_seed(seed)
    W = torch.randn(24, 48)

    # Correlated inputs make the Hessian off-diagonal terms meaningful.
    Z = torch.randn(192, 48)
    mixing = torch.eye(48) + 0.15 * torch.randn(48, 48)
    X = Z @ mixing

    naive = naive_round(W, n_bits=4)
    gptq, _ = gptq_columnwise(W, X, n_bits=4)
    return {
        "naive_mse": reconstruction_mse(W, naive, X),
        "gptq_mse": reconstruction_mse(W, gptq, X),
        "compression_bits": 4.0,
    }


def _self_test() -> None:
    torch.manual_seed(0)
    W = torch.randn(8, 16)
    X = torch.randn(64, 16)

    Q, E = gptq_columnwise(W, X, n_bits=4)
    assert Q.shape == W.shape
    assert E.shape == W.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(E).all()

    result = toy_comparison(seed=0)
    assert result["gptq_mse"] <= result["naive_mse"] * 1.05

    for key, value in result.items():
        print(f"{key}: {value}")
    print("gptq_original_minimal self-test passed")


if __name__ == "__main__":
    _self_test()
