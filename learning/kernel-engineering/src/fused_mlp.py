"""Fused MLP — GeLU + matmul fusion saves HBM round-trip."""
from __future__ import annotations
import math


def gelu(x: float) -> float:
    """Approximate GeLU."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def mlp_unfused(x: list[list[float]], W1: list[list[float]],
                W2: list[list[float]]) -> list[list[float]]:
    """3 HBM round-trips: matmul1 → write h, read h → gelu → write, read → matmul2."""
    N, D = len(x), len(x[0])
    H = len(W1[0])
    h = [[sum(x[i][d] * W1[d][k] for d in range(D)) for k in range(H)] for i in range(N)]
    h2 = [[gelu(v) for v in row] for row in h]
    out = [[sum(h2[i][k] * W2[k][d] for k in range(H)) for d in range(D)] for i in range(N)]
    return out


def mlp_fused(x: list[list[float]], W1: list[list[float]],
              W2: list[list[float]]) -> list[list[float]]:
    """Fused: per-row keep activation in registers, never write h to HBM."""
    N, D = len(x), len(x[0])
    H = len(W1[0])
    out = [[0.0] * D for _ in range(N)]
    for i in range(N):
        # Compute the row of activations in registers
        h_row = [gelu(sum(x[i][d] * W1[d][k] for d in range(D))) for k in range(H)]
        for d_out in range(D):
            out[i][d_out] = sum(h_row[k] * W2[k][d_out] for k in range(H))
    return out


def hbm_traffic(N: int, D: int, H: int, fused: bool, dtype_bytes: int = 2) -> int:
    """Bytes transferred to/from HBM."""
    weights = dtype_bytes * (D * H + H * D)        # W1 + W2 read once
    io = dtype_bytes * (N * D + N * D)             # x read, out write
    if fused:
        return weights + io
    # Unfused: hidden h written + read once (2× N*H)
    return weights + io + 2 * dtype_bytes * N * H


def _self_test() -> None:
    import random
    random.seed(13)
    N, D, H = 4, 6, 12
    x = [[random.random() for _ in range(D)] for _ in range(N)]
    W1 = [[random.random() for _ in range(H)] for _ in range(D)]
    W2 = [[random.random() for _ in range(D)] for _ in range(H)]
    a = mlp_unfused(x, W1, W2)
    b = mlp_fused(x, W1, W2)
    for i in range(N):
        for d in range(D):
            assert abs(a[i][d] - b[i][d]) < 1e-9

    unfused = hbm_traffic(2048, 4096, 16384, fused=False)
    fused = hbm_traffic(2048, 4096, 16384, fused=True)
    savings = (unfused - fused) / unfused
    assert savings > 0.0, savings
    print(f"[OK] fused_mlp (saved {savings * 100:.1f}% HBM)")


if __name__ == "__main__":
    _self_test()
