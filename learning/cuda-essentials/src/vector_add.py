"""Vector add: the "hello world" of CUDA kernels."""
from __future__ import annotations
from common import launch_config


def vector_add_kernel(a: list[float], b: list[float], c: list[float],
                       n: int, tid: int) -> None:
    """Simulated per-thread kernel body."""
    if tid < n:
        c[tid] = a[tid] + b[tid]


def launch_vector_add(a: list[float], b: list[float]) -> list[float]:
    n = len(a)
    assert len(b) == n
    c = [0.0] * n
    grid = launch_config(n, threads_per_block=256)
    # Simulate launch: iterate over global thread index
    for bid in range(grid.n_blocks()):
        for local_tid in range(grid.block_dim[0]):
            global_tid = bid * grid.block_dim[0] + local_tid
            vector_add_kernel(a, b, c, n, global_tid)
    return c


def _self_test() -> None:
    a = list(range(1000))
    b = [2.0 * x for x in a]
    c = launch_vector_add(a, b)
    assert c[42] == 3.0 * 42
    assert c[-1] == 3.0 * 999
    # n not multiple of block size
    a2 = list(range(513))
    c2 = launch_vector_add(a2, a2)
    assert len(c2) == 513
    assert c2[512] == 1024.0
    print("[OK] vector_add")


if __name__ == "__main__":
    _self_test()
