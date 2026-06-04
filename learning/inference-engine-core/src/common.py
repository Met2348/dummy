"""Shared abstractions for inference engine experiments.

A Request flows through prefill -> decode iterations.  Each request owns its
KV cache slice (or block list in PagedAttention).  Throughput = sum(out_tokens)
/ elapsed.  Latency = per-token decode time (decode iter wallclock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class Request:
    """One inference request, identified by integer id."""
    rid: int
    prompt_ids: List[int]
    max_new_tokens: int
    output_ids: List[int] = field(default_factory=list)
    finished: bool = False
    # Index into shared KV cache (naive) or list of block ids (paged)
    cache_ref: object = None
    # For metrics
    t_admit: float = 0.0
    t_first_token: float = 0.0
    t_finish: float = 0.0

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class EngineMetrics:
    n_requests: int = 0
    n_tokens_in: int = 0
    n_tokens_out: int = 0
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(self.t_end - self.t_start, 1e-9)

    @property
    def throughput_out(self) -> float:
        return self.n_tokens_out / self.elapsed

    @property
    def throughput_total(self) -> float:
        return (self.n_tokens_in + self.n_tokens_out) / self.elapsed

    def report(self) -> str:
        return (
            f"reqs={self.n_requests} in={self.n_tokens_in} out={self.n_tokens_out} "
            f"elapsed={self.elapsed:.2f}s "
            f"out_tps={self.throughput_out:.1f} total_tps={self.throughput_total:.1f}"
        )


def now() -> float:
    return time.perf_counter()


def stub_token_step(req: Request) -> int:
    """Deterministic mock token generator for engine smoke tests."""
    return (req.prompt_ids[0] + len(req.output_ids)) % 50257
