"""Multi-model router — pick model tier based on query complexity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelTier:
    name: str
    cost_per_million: float       # USD
    capability: float             # 0-1 score


TIERS = [
    ModelTier("small-1b", cost_per_million=0.05, capability=0.4),
    ModelTier("medium-7b", cost_per_million=0.30, capability=0.7),
    ModelTier("large-70b", cost_per_million=2.00, capability=0.9),
    ModelTier("thinking-o3", cost_per_million=10.0, capability=0.98),
]


def heuristic_complexity(query: str) -> float:
    """Crude complexity score: long queries + math/code keywords lift it."""
    score = 0.0
    score += min(1.0, len(query) / 500)
    keywords = ["prove", "derive", "complex", "step-by-step", "algorithm", "optimise"]
    score += 0.2 * sum(1 for k in keywords if k.lower() in query.lower())
    return min(score, 1.0)


def route(query: str) -> ModelTier:
    c = heuristic_complexity(query)
    if c < 0.25:
        return TIERS[0]
    if c < 0.55:
        return TIERS[1]
    if c < 0.85:
        return TIERS[2]
    return TIERS[3]


def estimate_total_cost(queries: List[str], tokens_per_query: int = 500) -> Dict[str, float]:
    by_tier: Dict[str, float] = {t.name: 0.0 for t in TIERS}
    for q in queries:
        t = route(q)
        by_tier[t.name] += tokens_per_query / 1_000_000 * t.cost_per_million
    return by_tier
