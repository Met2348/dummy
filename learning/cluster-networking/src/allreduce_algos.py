"""All-reduce algorithms: ring / tree / 2D-mesh / halving-doubling."""
from __future__ import annotations
from common import Link


def ring_allreduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Bandwidth-optimal. T = 2(N-1)/N * size/BW. Latency = 2(N-1) hops."""
    bw_per_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    n_steps = 2 * (n_gpus - 1)
    return n_steps * (link.latency_us + bw_per_step)


def tree_allreduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Latency-optimal (log N). T = 2 * log2(N) * (lat + size/BW)."""
    import math
    n_steps = 2 * int(math.ceil(math.log2(n_gpus)))
    bw_step = (bytes_total / link.bw_gb_s) / 1e9 * 1e6
    return n_steps * (link.latency_us + bw_step / 2)


def halving_doubling(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Rabenseifner: reduce-scatter halving + all-gather doubling. Best for big msg."""
    import math
    n_steps = 2 * int(math.ceil(math.log2(n_gpus)))
    bw_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return n_steps * (link.latency_us + bw_step)


def pick_algorithm(n_gpus: int, bytes_total: int, link: Link) -> str:
    """NCCL-like crossover heuristic."""
    if bytes_total < 1024:           # tiny: tree
        return "tree"
    if n_gpus <= 8:                  # NVLink ring is king
        return "ring"
    return "halving_doubling"


def _self_test() -> None:
    from common import LINKS
    nvl = LINKS["nvlink4"]
    ib = LINKS["ib_ndr"]
    # 1 GB on 8 GPUs over NVLink should be very fast.
    r = ring_allreduce(8, int(1e9), nvl)
    assert r < 50_000, r       # <50 ms
    # Latency-bound regime (1 KB)
    tiny_ring = ring_allreduce(64, 1024, ib)
    tiny_tree = tree_allreduce(64, 1024, ib)
    assert tiny_tree < tiny_ring, (tiny_ring, tiny_tree)

    # Algorithm picker
    assert pick_algorithm(8, int(1e9), nvl) == "ring"
    assert pick_algorithm(64, int(1e9), ib) == "halving_doubling"
    assert pick_algorithm(64, 256, ib) == "tree"
    print(f"[OK] allreduce_algos (8-GPU 1GB ring {r:.0f}us)")


if __name__ == "__main__":
    _self_test()
