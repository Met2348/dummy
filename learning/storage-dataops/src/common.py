"""Storage layers: local NVMe, Lustre, S3, and RAM."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Storage:
    name: str
    read_gb_s: float
    write_gb_s: float
    iops_random: int
    latency_us: float
    cap_pb: float


TIERS = {
    "ram":        Storage("Host DDR5",      80.0,  80.0, 10_000_000,    0.1, 0.002),
    "nvme_pcie5": Storage("NVMe Gen5",      14.0,  12.0,  2_000_000,   80.0, 0.030),
    "nvme_raid":  Storage("NVMe RAID0 8x", 100.0,  80.0,  8_000_000,   80.0, 0.250),
    "lustre":     Storage("Lustre OSS pool", 500.0, 400.0,   500_000,  500.0, 20.0),
    "gpfs":       Storage("GPFS DSS pool",   600.0, 500.0,   800_000,  400.0, 50.0),
    "s3":         Storage("S3 (regional)",     1.0,   1.0,     10_000, 50000.0, 1000.0),
}


def time_to_read(bytes_total: int, tier: Storage) -> float:
    """Seconds, including round-trip latency."""
    transfer_s = (bytes_total / 1e9) / tier.read_gb_s
    return tier.latency_us / 1e6 + transfer_s


def _self_test() -> None:
    # 100 GB shard from Lustre is about 0.2 s plus 0.5 ms.
    t = time_to_read(int(100e9), TIERS["lustre"])
    assert 0.15 < t < 0.25, t
    # Same from S3 is about 100 s plus 50 ms.
    t_s3 = time_to_read(int(100e9), TIERS["s3"])
    assert t_s3 > 200 * t, (t, t_s3)
    print(f"[OK] storage_dataops.common (100GB Lustre {t:.2f}s, S3 {t_s3:.0f}s)")


if __name__ == "__main__":
    _self_test()
