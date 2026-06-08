"""SM occupancy calculator: H100 132 SM, 2048 threads/SM."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SMLimits:
    max_threads_per_sm: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    max_regs_per_sm: int
    max_smem_per_sm_kb: int


H100_SM = SMLimits(2048, 64, 32, 65536, 228)


def occupancy(threads_per_block: int, regs_per_thread: int,
              smem_per_block_kb: float, limits: SMLimits = H100_SM) -> dict:
    blk_by_threads = limits.max_threads_per_sm // threads_per_block
    blk_by_warps = limits.max_warps_per_sm // ((threads_per_block + 31) // 32)
    blk_by_regs = (limits.max_regs_per_sm // (regs_per_thread * threads_per_block)
                   if regs_per_thread > 0 else 999)
    blk_by_smem = (int(limits.max_smem_per_sm_kb // smem_per_block_kb)
                   if smem_per_block_kb > 0 else 999)
    blk_by_max = limits.max_blocks_per_sm
    blocks = min(blk_by_threads, blk_by_warps, blk_by_regs, blk_by_smem, blk_by_max)
    active_warps = blocks * ((threads_per_block + 31) // 32)
    occ = active_warps / limits.max_warps_per_sm
    bottleneck = min(
        [("threads", blk_by_threads), ("warps", blk_by_warps),
         ("regs", blk_by_regs), ("smem", blk_by_smem), ("max_blocks", blk_by_max)],
        key=lambda x: x[1])[0]
    return {"blocks_per_sm": blocks, "occupancy": round(occ, 3),
            "active_warps": active_warps, "bottleneck": bottleneck}


def _self_test() -> None:
    # Good kernel: 256 threads, 32 regs, 16 KB SMEM gives high occupancy.
    good = occupancy(256, 32, 16.0)
    assert good["occupancy"] >= 0.5, good
    # Register-hungry: 256 threads, 128 regs gives a register bottleneck.
    bad_regs = occupancy(256, 128, 16.0)
    assert bad_regs["bottleneck"] == "regs", bad_regs
    # SMEM-hungry: 100 KB SMEM gives an smem bottleneck.
    bad_smem = occupancy(256, 32, 100.0)
    assert bad_smem["bottleneck"] == "smem", bad_smem
    print(f"[OK] sm_occupancy (good kernel occ {good['occupancy']:.2f})")


if __name__ == "__main__":
    _self_test()
