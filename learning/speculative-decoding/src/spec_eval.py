"""Evaluation helpers: accept rate, MAU, sim speedup."""
from __future__ import annotations

from dataclasses import dataclass
from common import SpecMetrics


def compute_mau(m: SpecMetrics) -> float:
    return m.mau


def compute_accept_rate(m: SpecMetrics) -> float:
    return m.accept_rate


def sim_speedup(m: SpecMetrics, draft_cost_ratio: float = 0.1) -> float:
    """Simulated speedup vs baseline of 1 token / iter.

    cost = draft_cost_ratio * drafted + 1 verify (large) per iter
    baseline = 1 large per token.
    Total iters in baseline = n_tokens_out; per iter cost = 1 (large).
    Speedup = n_tokens_out / (m.n_iters * (1 + draft_cost_ratio * k))
    where k = drafted / iters.
    """
    if m.n_iters == 0:
        return 1.0
    k_per_iter = m.n_drafted / m.n_iters
    iter_cost = 1 + draft_cost_ratio * k_per_iter
    return m.n_tokens_out / (m.n_iters * iter_cost)
