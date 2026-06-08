"""SHARP (in-network reduction): Mellanox/Nvidia switch offload."""
from __future__ import annotations
from common import Link


def sharp_allreduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """In-network reduction at the switch: 2-step constant cost.

    Each GPU sends size/N up to switch, switch aggregates, multicasts result.
    Total traffic per GPU = 2 * size/N (one up, one down).
    Latency O(1) instead of O(log N) or O(N).
    """
    bw_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return 2 * (link.latency_us + bw_step)


def speedup_vs_ring(n_gpus: int, bytes_total: int, link: Link) -> float:
    from allreduce_algos import ring_allreduce
    r = ring_allreduce(n_gpus, bytes_total, link)
    s = sharp_allreduce(n_gpus, bytes_total, link)
    return r / s


def _self_test() -> None:
    from common import LINKS
    ib = LINKS["ib_ndr"]
    # 64-GPU 100 MB: SHARP should win.
    sp = speedup_vs_ring(64, int(1e8), ib)
    assert sp > 1.0, sp
    # Larger N gives a bigger win because ring has 2*(N-1) hops and SHARP stays 2.
    sp_big = speedup_vs_ring(512, int(1e8), ib)
    assert sp_big > sp, (sp, sp_big)
    print(f"[OK] sharp_inline (64-GPU speedup {sp:.1f}x, 512-GPU {sp_big:.1f}x)")


if __name__ == "__main__":
    _self_test()
