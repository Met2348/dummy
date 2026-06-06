"""NCCL collective primitives — mock latency/BW model."""
from __future__ import annotations
from common import Link
from allreduce_algos import ring_allreduce, tree_allreduce


def all_gather(n_gpus: int, bytes_per_gpu: int, link: Link) -> float:
    """T = (N-1)/N * total_size / BW. Bandwidth-optimal ring."""
    total = bytes_per_gpu * n_gpus
    bw_per_step = (total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return (n_gpus - 1) * (link.latency_us + bw_per_step)


def reduce_scatter(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Same cost as all_gather."""
    bw_per_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return (n_gpus - 1) * (link.latency_us + bw_per_step)


def all_reduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Ring all_reduce = reduce_scatter + all_gather."""
    return ring_allreduce(n_gpus, bytes_total, link)


def broadcast(n_gpus: int, bytes_total: int, link: Link) -> float:
    """log(N) tree broadcast."""
    import math
    n_steps = int(math.ceil(math.log2(n_gpus)))
    bw_step = (bytes_total / link.bw_gb_s) / 1e9 * 1e6
    return n_steps * (link.latency_us + bw_step)


def all_to_all(n_gpus: int, bytes_per_pair: int, link: Link) -> float:
    """Each GPU sends bytes_per_pair to N-1 others. Saturates per-GPU egress BW."""
    total_egress = (n_gpus - 1) * bytes_per_pair
    return link.latency_us + (total_egress / link.bw_gb_s) / 1e9 * 1e6


def _self_test() -> None:
    from common import LINKS
    nvl = LINKS["nvlink4"]
    # all_reduce == reduce_scatter + all_gather (cost relation)
    rs = reduce_scatter(8, int(1e8), nvl)
    ag = all_gather(8, int(1e8) // 8, nvl)
    ar = all_reduce(8, int(1e8), nvl)
    assert abs(ar - (rs + ag)) / ar < 0.05, (rs, ag, ar)

    # broadcast log scaling
    bc8 = broadcast(8, int(1e7), nvl)
    bc16 = broadcast(16, int(1e7), nvl)
    bc64 = broadcast(64, int(1e7), nvl)
    assert bc8 < bc16 < bc64
    # ratio close to log2 ratio
    import math
    assert bc64 / bc8 < math.log2(64) / math.log2(8) + 0.5
    print(f"[OK] nccl_collectives (8-GPU AR={ar:.0f}us = RS+AG)")


if __name__ == "__main__":
    _self_test()
