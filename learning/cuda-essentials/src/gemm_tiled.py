"""Tiled GEMM: the canonical CUDA optimization story."""
from __future__ import annotations


def gemm_naive(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Each output element computed by 1 thread, K loop in-thread, no reuse."""
    M, K = len(A), len(A[0])
    K2, N = len(B), len(B[0])
    assert K == K2
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def gemm_tiled(A: list[list[float]], B: list[list[float]], tile: int = 16) -> list[list[float]]:
    """Block-tiled GEMM with shared-memory-style reuse per K step."""
    M, K = len(A), len(A[0])
    _, N = len(B), len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for ii in range(0, M, tile):
        for jj in range(0, N, tile):
            for kk in range(0, K, tile):
                # Simulate SMEM load + reuse
                for i in range(ii, min(ii + tile, M)):
                    for j in range(jj, min(jj + tile, N)):
                        s = C[i][j]
                        for k in range(kk, min(kk + tile, K)):
                            s += A[i][k] * B[k][j]
                        C[i][j] = s
    return C


def hbm_traffic_naive(M: int, N: int, K: int, dtype_bytes: int = 4) -> int:
    """Each thread loads K elements from A and K from B."""
    return dtype_bytes * M * N * 2 * K


def hbm_traffic_tiled(M: int, N: int, K: int, tile: int, dtype_bytes: int = 4) -> int:
    """A and B tiles loaded once per (ii,jj,kk) sub-block."""
    n_tiles_M = (M + tile - 1) // tile
    n_tiles_N = (N + tile - 1) // tile
    n_tiles_K = (K + tile - 1) // tile
    a_loads = n_tiles_M * n_tiles_N * n_tiles_K * tile * tile
    b_loads = n_tiles_M * n_tiles_N * n_tiles_K * tile * tile
    return dtype_bytes * (a_loads + b_loads)


def _self_test() -> None:
    import random
    random.seed(0)
    M, N, K = 32, 32, 32
    A = [[random.random() for _ in range(K)] for _ in range(M)]
    B = [[random.random() for _ in range(N)] for _ in range(K)]
    C1 = gemm_naive(A, B)
    C2 = gemm_tiled(A, B, tile=8)
    for i in range(M):
        for j in range(N):
            assert abs(C1[i][j] - C2[i][j]) < 1e-9, (i, j)

    # Traffic comparison on a larger problem.
    naive_b = hbm_traffic_naive(1024, 1024, 1024)
    tiled_b = hbm_traffic_tiled(1024, 1024, 1024, tile=32)
    speedup = naive_b / tiled_b
    assert speedup > 16.0, f"expected >16x reduction, got {speedup:.1f}"
    print(f"[OK] gemm_tiled (HBM traffic {speedup:.1f}x reduction)")


if __name__ == "__main__":
    _self_test()
