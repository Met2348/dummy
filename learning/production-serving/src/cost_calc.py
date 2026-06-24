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


def demo() -> None:
    print("=== LLM 服务成本计算 ===")
    selfhost = Deployment("self-host 8xA100", gpu_cost_per_hour=2.0, tok_per_s_per_gpu=2500, n_gpus=8)
    w = Workload(qps=20, avg_in_tokens=500, avg_out_tokens=150)
    print(f"$/M tokens (self-host)   : ${dollars_per_million_tokens(selfhost):.3f}")
    print(f"该 workload 每小时成本    : ${cost_for_workload(selfhost, w):.2f}")
    print(f"40% 前缀缓存命中每小时省  : ${cache_hit_savings(selfhost, w, hit_rate=0.4):.2f}")
    small = Deployment("small-0.5B", gpu_cost_per_hour=1.0, tok_per_s_per_gpu=8000)
    large = Deployment("large-70B", gpu_cost_per_hour=16.0, tok_per_s_per_gpu=600)
    for c in (0.2, 0.8):
        print(f"query 复杂度 {c} -> 路由到 {cost_aware_route(c, small, large).name}")


if __name__ == "__main__":
    demo()
