"""Reduce kernel: naive vs Brent's vs warp-shuffle."""
from __future__ import annotations
from warp_primitives import warp_reduce_sum, WARP_SIZE


def reduce_naive(data: list[float]) -> float:
    """Sequential baseline with no parallelism."""
    s = 0.0
    for x in data:
        s += x
    return s


def reduce_brent_kung(data: list[float]) -> float:
    """Tree reduction with log2 steps, each halving active threads."""
    v = list(data)
    stride = 1
    while stride < len(v):
        i = 0
        while i + stride < len(v):
            v[i] += v[i + stride]
            i += 2 * stride
        stride *= 2
    return v[0]


def reduce_warp_shuffle(data: list[float]) -> float:
    """Per-warp shuffle reduce then sum partials."""
    partials = []
    for w_start in range(0, len(data), WARP_SIZE):
        chunk = data[w_start:w_start + WARP_SIZE]
        if len(chunk) < WARP_SIZE:
            chunk = chunk + [0] * (WARP_SIZE - len(chunk))
        partials.append(warp_reduce_sum([int(x) for x in chunk]))
    return float(sum(partials))


def _self_test() -> None:
    data = [float(i) for i in range(1024)]
    expected = sum(data)
    assert abs(reduce_naive(data) - expected) < 1e-3
    assert abs(reduce_brent_kung(data) - expected) < 1e-3
    assert abs(reduce_warp_shuffle(data) - expected) < 1e-3
    print(f"[OK] reduce_kernel (sum 1024 = {int(expected)})")


if __name__ == "__main__":
    _self_test()
