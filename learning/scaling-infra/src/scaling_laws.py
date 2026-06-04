"""Scaling laws 工具：Chinchilla / Kaplan / IsoFLOP."""
from __future__ import annotations


def chinchilla_loss(N: int, D: int,
                    A: float = 406.4, B: float = 410.7, E: float = 1.69,
                    alpha: float = 0.34, beta: float = 0.28) -> float:
    """Hoffmann 2022 公式."""
    return E + A * (N ** -alpha) + B * (D ** -beta)


def kaplan_loss(N: int, D: int = None) -> float:
    """Kaplan 2020 简化."""
    Nc = 8.8e13
    alpha = 0.076
    return (Nc / N) ** alpha


def compute_flops(N: int, D: int) -> float:
    """6 N D 是 Chinchilla 常用估计."""
    return 6 * N * D


def chinchilla_optimal_split(C: float) -> tuple:
    """给 budget C, 求最优 (N, D).

    论文得: D / N ≈ 20.
    """
    N_opt_ratio = 20.0
    N = (C / (6 * N_opt_ratio)) ** 0.5
    D = C / (6 * N)
    return int(N), int(D)


def over_train_split(C: float, ratio: float = 200.0) -> tuple:
    """Llama-3 风格 over-train, ratio=D/N."""
    N = (C / (6 * ratio)) ** 0.5
    D = N * ratio
    return int(N), int(D)


if __name__ == "__main__":
    print("=== Chinchilla vs Llama-3 给定 C = 1e23 FLOPs ===")
    C = 1e23

    N1, D1 = chinchilla_optimal_split(C)
    L1 = chinchilla_loss(N1, D1)
    print(f"Chinchilla 1:20  N={N1/1e9:.1f}B D={D1/1e9:.1f}B loss={L1:.4f}")

    N2, D2 = over_train_split(C, ratio=200)
    L2 = chinchilla_loss(N2, D2)
    print(f"Over-train 1:200 N={N2/1e9:.1f}B D={D2/1e9:.1f}B loss={L2:.4f}")

    print("\n=== 表 ===")
    for ratio in [10, 20, 50, 100, 200, 500]:
        N, D = over_train_split(C, ratio)
        L = chinchilla_loss(N, D)
        print(f"  ratio={ratio:>3} N={N/1e9:6.2f}B D={D/1e9:7.1f}B loss={L:.4f}")
