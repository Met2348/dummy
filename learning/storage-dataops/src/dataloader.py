"""Dataloader pipeline — fetch / decode / collate, with prefetch model."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    per_sample_us: float


def naive_pipeline(stages: list[Stage], n_samples: int) -> float:
    """Sequential: total = sum(stages) * N. Useful baseline."""
    per_sample = sum(s.per_sample_us for s in stages)
    return per_sample * n_samples


def pipelined(stages: list[Stage], n_samples: int, n_workers: int = 4) -> float:
    """N workers in parallel + double-buffering.

    Throughput = min over stages of (n_workers / per_sample_us).
    Total time = max(stages) * N (assuming compute is hidden by overlap).
    """
    parallel_per_sample = max(s.per_sample_us for s in stages) / n_workers
    return parallel_per_sample * n_samples


def bottleneck_stage(stages: list[Stage]) -> Stage:
    return max(stages, key=lambda s: s.per_sample_us)


def _self_test() -> None:
    stages = [
        Stage("fetch_lustre", 500),
        Stage("decode_jpeg", 2000),
        Stage("augment", 800),
        Stage("collate", 100),
    ]
    naive = naive_pipeline(stages, 1000)        # = 3400 ms
    fast = pipelined(stages, 1000, n_workers=4)  # = 2000/4 * 1000 = 500 ms
    speedup = naive / fast
    assert speedup > 5.0, speedup
    bn = bottleneck_stage(stages)
    assert bn.name == "decode_jpeg"
    print(f"[OK] dataloader (speedup {speedup:.1f}x, bottleneck={bn.name})")


if __name__ == "__main__":
    _self_test()
