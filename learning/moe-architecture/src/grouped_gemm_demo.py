"""Grouped GEMM 教学版 — megablocks 风格."""
from __future__ import annotations

import time

import torch


def grouped_gemm_naive(xs: list[torch.Tensor],
                       ws: list[torch.Tensor]) -> list[torch.Tensor]:
    """Naive: 循环每 expert 单独 GEMM."""
    return [x @ w for x, w in zip(xs, ws)]


def grouped_gemm_fused(xs: list[torch.Tensor],
                       ws: list[torch.Tensor]) -> torch.Tensor:
    """Block-diagonal fuse: 把所有 GEMM 拼一次矩阵乘."""
    X = torch.block_diag(*xs)
    W = torch.block_diag(*ws)
    return X @ W


def bench():
    n_expert = 8
    n_tok_per = 32
    d_in = 64
    d_out = 64
    xs = [torch.randn(n_tok_per, d_in) for _ in range(n_expert)]
    ws = [torch.randn(d_in, d_out) for _ in range(n_expert)]

    t0 = time.time()
    for _ in range(100):
        out_naive = grouped_gemm_naive(xs, ws)
    t_naive = time.time() - t0

    t0 = time.time()
    for _ in range(100):
        out_fused = grouped_gemm_fused(xs, ws)
    t_fused = time.time() - t0

    print(f"naive:  {t_naive*10:.2f} ms / iter")
    print(f"fused:  {t_fused*10:.2f} ms / iter")
    print(f"speedup: {t_naive / t_fused:.2f}×")


if __name__ == "__main__":
    bench()
