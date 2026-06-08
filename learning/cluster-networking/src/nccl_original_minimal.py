"""Paper-shaped NCCL model for protocols, channels, and algorithms.

This is a teaching model for the NCCL analysis paper. It keeps only the
decisions that the paper repeatedly returns to: protocol choice, channel
count, and ring-vs-tree behavior under message-size and topology changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from common import Link, LINKS


@dataclass(frozen=True)
class ProtocolProfile:
    name: str
    per_hop_latency_us: float
    bandwidth_fraction: float
    data_bytes_per_unit: int
    flag_bytes_per_unit: int
    allows_gdr_rdma: bool
    note: str

    @property
    def payload_fraction(self) -> float:
        total = self.data_bytes_per_unit + self.flag_bytes_per_unit
        if total == 0:
            return 1.0
        return self.data_bytes_per_unit / total


SIMPLE = ProtocolProfile(
    name="Simple",
    per_hop_latency_us=6.0,
    bandwidth_fraction=0.98,
    data_bytes_per_unit=512 * 1024,
    flag_bytes_per_unit=0,
    allows_gdr_rdma=True,
    note="large chunks, memory fences, near-peak bandwidth",
)

LL = ProtocolProfile(
    name="LL",
    per_hop_latency_us=1.0,
    bandwidth_fraction=0.35,
    data_bytes_per_unit=4,
    flag_bytes_per_unit=4,
    allows_gdr_rdma=False,
    note="4B data plus 4B flag, low latency but low bandwidth",
)

LL128 = ProtocolProfile(
    name="LL128",
    per_hop_latency_us=2.0,
    bandwidth_fraction=0.95,
    data_bytes_per_unit=120,
    flag_bytes_per_unit=8,
    allows_gdr_rdma=True,
    note="120B data plus 8B flag, needs ordered 128-byte writes",
)


def protocol_transfer_time_us(
    bytes_total: int,
    link: Link,
    protocol: ProtocolProfile,
    hops: int = 1,
) -> float:
    """Estimate transfer time with protocol overhead and per-hop latency."""
    if bytes_total < 0:
        raise ValueError("bytes_total must be non-negative")
    if hops < 1:
        raise ValueError("hops must be at least 1")

    effective_bw = link.bw_gb_s * protocol.bandwidth_fraction * protocol.payload_fraction
    transfer_us = (bytes_total / 1e9) / effective_bw * 1e6 if effective_bw else 0.0
    return hops * protocol.per_hop_latency_us + transfer_us


def choose_protocol(
    bytes_total: int,
    *,
    intra_node: bool,
    supports_ll128: bool = True,
) -> ProtocolProfile:
    """Mimic the paper's qualitative NCCL protocol crossover."""
    if bytes_total < 64 * 1024:
        if supports_ll128:
            return LL128 if intra_node else LL
        return LL

    if intra_node and supports_ll128 and bytes_total < 32 * 1024 * 1024:
        return LL128

    return SIMPLE


def choose_channels(
    bytes_total: int,
    *,
    max_channels: int = 16,
    fifo_bytes: int = 512 * 1024,
) -> int:
    """Reduce channels when each channel would underfill the NIC FIFO."""
    if bytes_total <= 0:
        return 1
    if max_channels < 1:
        raise ValueError("max_channels must be positive")

    channels = min(max_channels, max(1, math.ceil(bytes_total / fifo_bytes)))
    while channels > 1 and bytes_total / channels < fifo_bytes:
        channels -= 1
    return channels


def ring_steps(n_gpus: int) -> int:
    if n_gpus < 2:
        raise ValueError("n_gpus must be at least 2")
    return 2 * (n_gpus - 1)


def tree_steps(n_gpus: int) -> int:
    if n_gpus < 2:
        raise ValueError("n_gpus must be at least 2")
    return 2 * int(math.ceil(math.log2(n_gpus)))


def choose_algorithm(n_gpus: int, bytes_total: int) -> str:
    """Tree is latency-friendly; ring is bandwidth-friendly for large payloads."""
    if bytes_total < 1 * 1024 * 1024:
        return "tree"
    if n_gpus <= 8 and bytes_total < 16 * 1024 * 1024:
        return "tree"
    return "ring"


def allreduce_plan(
    n_gpus: int,
    bytes_total: int,
    link: Link,
    *,
    intra_node: bool,
    supports_ll128: bool = True,
    max_channels: int = 16,
) -> dict:
    """Return a compact all-reduce plan in the vocabulary of the paper."""
    protocol = choose_protocol(
        bytes_total,
        intra_node=intra_node,
        supports_ll128=supports_ll128,
    )
    algorithm = choose_algorithm(n_gpus, bytes_total)
    steps = tree_steps(n_gpus) if algorithm == "tree" else ring_steps(n_gpus)
    channels = choose_channels(bytes_total, max_channels=max_channels)
    per_channel_bytes = math.ceil(bytes_total / channels)
    per_channel_time = protocol_transfer_time_us(
        per_channel_bytes,
        link,
        protocol,
        hops=steps,
    )
    return {
        "algorithm": algorithm,
        "protocol": protocol.name,
        "channels": channels,
        "steps": steps,
        "per_channel_bytes": per_channel_bytes,
        "estimated_time_us": per_channel_time,
        "protocol_note": protocol.note,
    }


def _self_test() -> None:
    nvl = LINKS["nvlink4"]
    roce = LINKS["roce_400g"]

    tiny = 16 * 1024
    huge = 512 * 1024 * 1024

    assert choose_protocol(tiny, intra_node=False).name == "LL"
    assert choose_protocol(tiny, intra_node=True).name == "LL128"
    assert choose_protocol(huge, intra_node=False).name == "Simple"

    tiny_ll = protocol_transfer_time_us(tiny, roce, LL)
    tiny_simple = protocol_transfer_time_us(tiny, roce, SIMPLE)
    assert tiny_ll < tiny_simple, (tiny_ll, tiny_simple)

    large_simple = protocol_transfer_time_us(huge, roce, SIMPLE, hops=16)
    large_ll = protocol_transfer_time_us(huge, roce, LL, hops=16)
    assert large_simple < large_ll, (large_simple, large_ll)

    assert choose_channels(128 * 1024, max_channels=16) == 1
    assert choose_channels(64 * 1024 * 1024, max_channels=16) == 16

    assert tree_steps(64) < ring_steps(64)
    assert choose_algorithm(64, 8 * 1024) == "tree"
    assert choose_algorithm(64, huge) == "ring"

    plan = allreduce_plan(64, huge, roce, intra_node=False)
    assert plan["algorithm"] == "ring"
    assert plan["protocol"] == "Simple"
    assert plan["channels"] == 16

    intra_plan = allreduce_plan(8, 8 * 1024 * 1024, nvl, intra_node=True)
    assert intra_plan["protocol"] == "LL128"
    print(f"[OK] nccl_original_minimal ({plan['protocol']} {plan['algorithm']}, "
          f"{plan['channels']} channels)")


if __name__ == "__main__":
    _self_test()
