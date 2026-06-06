"""Warp-level primitives — shuffle, ballot, reduce."""
from __future__ import annotations

WARP_SIZE = 32


def shfl_down_sync(values: list[int], delta: int) -> list[int]:
    """__shfl_down_sync: lane i reads from lane (i + delta). Lanes >= 32-delta keep their own."""
    n = len(values)
    assert n == WARP_SIZE, f"need exactly 32 lanes, got {n}"
    out = list(values)
    for i in range(WARP_SIZE):
        if i + delta < WARP_SIZE:
            out[i] = values[i + delta]
    return out


def warp_reduce_sum(values: list[int]) -> int:
    """Tree reduction via shuffle — log2(32) = 5 steps."""
    v = list(values)
    delta = WARP_SIZE // 2
    while delta > 0:
        shifted = shfl_down_sync(v, delta)
        v = [v[i] + shifted[i] for i in range(WARP_SIZE)]
        delta //= 2
    return v[0]      # lane 0 holds the result


def ballot_sync(mask: list[bool]) -> int:
    """__ballot_sync: 32-bit mask of which lanes voted true."""
    assert len(mask) == WARP_SIZE
    result = 0
    for i in range(WARP_SIZE):
        if mask[i]:
            result |= (1 << i)
    return result


def _self_test() -> None:
    v = [1] * WARP_SIZE
    assert warp_reduce_sum(v) == 32

    v2 = list(range(WARP_SIZE))        # 0..31, sum = 496
    assert warp_reduce_sum(v2) == 496

    mask = [i % 2 == 0 for i in range(WARP_SIZE)]
    b = ballot_sync(mask)
    # bits 0, 2, 4, ... set
    assert bin(b).count("1") == 16
    assert b & 1 == 1
    assert b & 2 == 0
    print("[OK] warp_primitives (sum=496, ballot popcount=16)")


if __name__ == "__main__":
    _self_test()
