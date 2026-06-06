"""RMSNorm — fused root-mean-square normalization."""
from __future__ import annotations
import math


def rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """y = x / rms(x) * weight, where rms(x) = sqrt(mean(x²) + eps)."""
    n = len(x)
    s = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(s + eps)
    return [x[i] * inv * weight[i] for i in range(n)]


def rmsnorm_batch(X: list[list[float]], weight: list[float]) -> list[list[float]]:
    return [rmsnorm(row, weight) for row in X]


def fused_rmsnorm_linear(X: list[list[float]], norm_weight: list[float],
                          W: list[list[float]]) -> list[list[float]]:
    """Fuse RMSNorm + linear: never materialize normalized intermediate."""
    N, D = len(X), len(X[0])
    out_d = len(W[0])
    out = [[0.0] * out_d for _ in range(N)]
    for i in range(N):
        s = sum(v * v for v in X[i]) / D
        inv = 1.0 / math.sqrt(s + 1e-6)
        for j in range(out_d):
            acc = 0.0
            for d in range(D):
                acc += X[i][d] * inv * norm_weight[d] * W[d][j]
            out[i][j] = acc
    return out


def _self_test() -> None:
    import random
    random.seed(11)
    N, D, OD = 3, 8, 5
    X = [[random.random() for _ in range(D)] for _ in range(N)]
    w = [1.0] * D
    W = [[random.random() for _ in range(OD)] for _ in range(D)]

    # Unfused
    norm = rmsnorm_batch(X, w)
    Y1 = [[sum(norm[i][d] * W[d][j] for d in range(D)) for j in range(OD)] for i in range(N)]
    # Fused
    Y2 = fused_rmsnorm_linear(X, w, W)
    for i in range(N):
        for j in range(OD):
            assert abs(Y1[i][j] - Y2[i][j]) < 1e-9, (i, j)

    # Single-row sanity
    y = rmsnorm([1, 2, 3, 4], [1, 1, 1, 1])
    rms = math.sqrt((1 + 4 + 9 + 16) / 4)
    assert abs(y[0] - 1 / rms) < 1e-6
    print("[OK] rmsnorm_kernel")


if __name__ == "__main__":
    _self_test()
