"""$/M-token calculator + cache-hit savings + cost-aware routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Workload:
    qps: float
    avg_in_tokens: int
    avg_out_tokens: int


@dataclass
class Deployment:
    name: str
    gpu_cost_per_hour: float
    tok_per_s_per_gpu: float
    n_gpus: int = 1


def dollars_per_million_tokens(d: Deployment) -> float:
    cost_per_sec = d.gpu_cost_per_hour * d.n_gpus / 3600.0
    return cost_per_sec / d.tok_per_s_per_gpu * 1_000_000


def cost_for_workload(d: Deployment, w: Workload, hours: float = 1.0) -> float:
    total_tokens = (w.avg_in_tokens + w.avg_out_tokens) * w.qps * 3600.0 * hours
    return total_tokens / 1_000_000 * dollars_per_million_tokens(d)


def cache_hit_savings(d: Deployment, w: Workload, hit_rate: float, hit_discount: float = 0.9) -> float:
    """Estimate $ saved per hour when prefix cache hits cut input cost."""
    base = cost_for_workload(d, w, hours=1.0)
    saved_input_fraction = hit_rate * hit_discount * (w.avg_in_tokens / (w.avg_in_tokens + w.avg_out_tokens))
    return base * saved_input_fraction


def cost_aware_route(query_complexity: float, small: Deployment, large: Deployment, threshold: float = 0.5) -> Deployment:
    return large if query_complexity > threshold else small
