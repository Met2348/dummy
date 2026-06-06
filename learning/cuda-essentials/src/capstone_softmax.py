"""Capstone: numerically stable softmax kernel — naive → online (FlashAttn style)."""
from __future__ import annotations
import math


def softmax_naive(x: list[float]) -> list[float]:
    """3-pass: max, exp, normalize. Two HBM round-trips."""
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def softmax_online(x: list[float]) -> list[float]:
    """1-pass running max + running sum, FlashAttn-style.

    Invariant after step i:
      m_i = max(x[0..i])
      d_i = sum_{j<=i} exp(x[j] - m_i)
    Update rule:
      m_new = max(m_i, x[i+1])
      d_new = d_i * exp(m_i - m_new) + exp(x[i+1] - m_new)
    """
    m = -math.inf
    d = 0.0
    for v in x:
        m_new = max(m, v)
        d = d * math.exp(m - m_new) + math.exp(v - m_new)
        m = m_new
    return [math.exp(v - m) / d for v in x]


def _self_test() -> None:
    x = [3.0, 1.0, 0.2, -5.0, 7.0, 2.0]
    a = softmax_naive(x)
    b = softmax_online(x)
    for u, v in zip(a, b):
        assert abs(u - v) < 1e-9, (u, v)
    assert abs(sum(a) - 1.0) < 1e-9
    assert abs(sum(b) - 1.0) < 1e-9

    # Numerical stability — large values
    big = [1000.0, 1001.0, 1002.0]
    try:
        bn = softmax_naive(big)
        bo = softmax_online(big)
        for u, v in zip(bn, bo):
            assert abs(u - v) < 1e-9
    except OverflowError:
        raise AssertionError("naive should still work after subtracting max")
    print("[OK] capstone_softmax (online == naive)")


if __name__ == "__main__":
    _self_test()
