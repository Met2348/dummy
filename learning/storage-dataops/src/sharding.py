"""Dataset sharding strategies."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib


@dataclass
class Sample:
    sample_id: int
    bytes_size: int


def hash_shard(sample_id: int, n_shards: int) -> int:
    h = hashlib.sha1(str(sample_id).encode()).hexdigest()
    return int(h, 16) % n_shards


def range_shard(sample_id: int, n_shards: int, total: int) -> int:
    return min(sample_id * n_shards // total, n_shards - 1)


def round_robin_shard(sample_id: int, n_shards: int) -> int:
    return sample_id % n_shards


def shard_balance(strategy, samples: list[Sample], n_shards: int) -> dict:
    """Bytes per shard. Lower variance = better balance."""
    bytes_per = [0] * n_shards
    if strategy == "hash":
        for s in samples:
            bytes_per[hash_shard(s.sample_id, n_shards)] += s.bytes_size
    elif strategy == "range":
        for s in samples:
            bytes_per[range_shard(s.sample_id, n_shards, len(samples))] += s.bytes_size
    elif strategy == "round_robin":
        for s in samples:
            bytes_per[round_robin_shard(s.sample_id, n_shards)] += s.bytes_size

    mean = sum(bytes_per) / n_shards
    variance = sum((b - mean) ** 2 for b in bytes_per) / n_shards
    return {"per_shard": bytes_per, "mean": mean, "stddev": variance ** 0.5,
            "imbalance_pct": round(100 * (max(bytes_per) - mean) / mean, 1)}


def _self_test() -> None:
    # 10000 samples with skewed sizes: period-50 pattern, 100B..5000B repeating.
    samples = [Sample(i, 100 + (i % 50) * 100) for i in range(10000)]

    h = shard_balance("hash", samples, 8)
    r = shard_balance("range", samples, 8)
    rr = shard_balance("round_robin", samples, 8)

    # round_robin tied with hash for skewed data
    assert rr["imbalance_pct"] < 10.0, rr
    # NOTE: range lands at exactly 0.0% here -- not because range partitioning is
    # generally skew-resistant, but because this toy skew period (50) evenly divides
    # the per-shard sample count (10000/8=1250 -> exactly 25 full periods per shard),
    # so every contiguous range sees an identical size mix. Range only gets "unlucky"
    # (large imbalance) when the skew is *clustered by index* instead of periodic --
    # e.g. first 8000 samples at 100B + last 2000 at 5000B drives range imbalance to
    # ~363% while hash/round_robin both stay under 10% (verified empirically, not
    # exercised by this self-test's periodic data).
    print(f"[OK] sharding (hash imbalance {h['imbalance_pct']}%, "
          f"range {r['imbalance_pct']}%, rr {rr['imbalance_pct']}%)")


if __name__ == "__main__":
    _self_test()
