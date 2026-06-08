"""GPU memory hierarchy: registers, SMEM, L2, HBM."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MemTier:
    name: str
    size_per_sm_kb: float    # 0 means "per-GPU" not per-SM
    latency_cycles: int
    bandwidth_tb_s: float


H100_HIERARCHY = [
    MemTier("registers",      256.0, 1,    2000.0),   # 65k 32-bit regs/SM
    MemTier("shared_memory",  228.0, 30,    228.0),    # SMEM up to 228KB/SM
    MemTier("L1_cache",       128.0, 30,    128.0),
    MemTier("L2_cache",         0.0, 200,   12.0),     # 60MB shared
    MemTier("HBM3",             0.0, 600,   3.35),
]


def cost_to_load(bytes_needed: int, tier: MemTier) -> float:
    """Time in microseconds to move bytes_needed through tier."""
    return (bytes_needed / 1e9) / tier.bandwidth_tb_s * 1e6


def recommend_tier(working_set_bytes: int, reuse_count: int) -> MemTier:
    """Pick the lowest tier that fits the working set, given reuse."""
    if working_set_bytes <= 256 * 1024 and reuse_count > 10:
        return H100_HIERARCHY[0]
    if working_set_bytes <= 228 * 1024 and reuse_count > 2:
        return H100_HIERARCHY[1]
    if working_set_bytes <= 60 * 1024 * 1024:
        return H100_HIERARCHY[3]
    return H100_HIERARCHY[4]


def _self_test() -> None:
    # Small reused buffer: registers.
    t = recommend_tier(1024, reuse=100) if False else recommend_tier(1024, 100)
    assert t.name == "registers"
    # 100KB reused 5 times: SMEM.
    assert recommend_tier(100 * 1024, 5).name == "shared_memory"
    # 10MB: L2.
    assert recommend_tier(10 * 1024 * 1024, 1).name == "L2_cache"
    # 100MB: HBM.
    assert recommend_tier(100 * 1024 * 1024, 1).name == "HBM3"
    # Latency monotone increasing
    lats = [t.latency_cycles for t in H100_HIERARCHY]
    assert lats == sorted(lats)
    print("[OK] memory_hierarchy")


if __name__ == "__main__":
    _self_test()
