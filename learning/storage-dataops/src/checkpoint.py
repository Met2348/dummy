"""Checkpoint strategies: full, sharded, async, and DCP."""
from __future__ import annotations
from dataclasses import dataclass
from common import Storage


@dataclass
class CkptCost:
    sec: float
    bytes_written: int
    blocking: bool


def full_checkpoint(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """Rank 0 gathers all, writes once. Blocks training."""
    gather_s = model_bytes / 1e9 / 400 + 0.001     # NCCL gather BW ~400 GB/s aggregate
    write_s = model_bytes / 1e9 / tier.write_gb_s
    return CkptCost(gather_s + write_s, model_bytes, blocking=True)


def sharded_checkpoint(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """Each rank writes its shard, but the OSS pool's aggregate BW is the bottleneck."""
    write_s = model_bytes / 1e9 / tier.write_gb_s    # contention on tier
    return CkptCost(write_s, model_bytes, blocking=True)


def async_sharded(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """torch.distributed.checkpoint async: per-rank stage to host RAM via PCIe (independent).

    The blocking time is the PCIe stage, with each GPU independent.
    Background write to Lustre happens during compute and is not counted.
    """
    per_rank_bytes = model_bytes // n_gpus
    pcie_per_rank_gb_s = 32.0       # PCIe Gen5 x16 about 64 GB/s, halved.
    stage_s = per_rank_bytes / 1e9 / pcie_per_rank_gb_s
    return CkptCost(stage_s, model_bytes, blocking=False)


def _self_test() -> None:
    from common import TIERS
    lustre = TIERS["lustre"]
    # 70B model in BF16 = 140 GB
    model = int(140e9)
    n = 512

    f = full_checkpoint(model, n, lustre)
    s = sharded_checkpoint(model, n, lustre)
    a = async_sharded(model, n, lustre)

    assert f.blocking and s.blocking and not a.blocking
    assert s.sec < f.sec, (f, s)        # parallel write wins
    assert a.sec < s.sec, (s, a)        # async wins more
    print(f"[OK] checkpoint (full {f.sec:.2f}s, sharded {s.sec:.3f}s, async {a.sec:.3f}s)")


if __name__ == "__main__":
    _self_test()
