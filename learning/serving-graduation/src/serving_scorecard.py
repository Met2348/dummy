"""Graduation serving scorecard inspired by DistServe goodput.

DistServe asks a serving system to maximize request rate under latency SLOs.
For the graduation capstone, a "good" request also has to be correct. This
module turns the five-checkpoint demo report into a small operational scorecard:

* correctness gate
* TTFT-like latency gate
* TPOT-like per-token latency gate
* mock cost estimate
* effective goodput under an arrival rate
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class GraduationSLO:
    max_ttft_ms: float
    max_tpot_ms: float
    require_correct: bool = True


@dataclass(frozen=True)
class CandidateScore:
    ckpt: str
    correct: bool
    ttft_ms: float
    tpot_ms: float
    estimated_cost: float
    passes: bool


def _response_tokens(response: str) -> int:
    cleaned = response.replace("<think>", " ").replace("</think>", " ")
    return max(1, len(re.findall(r"[A-Za-z]+|\d+|\S", cleaned)))


def _mock_cost(size_mb: int, latency_ms: int) -> float:
    """A deterministic proxy: larger and slower checkpoints cost more."""
    return size_mb * latency_ms / 1_000_000.0


def score_report(report: Dict, slo: GraduationSLO) -> List[CandidateScore]:
    scores: List[CandidateScore] = []
    for row in report["results"]:
        tokens = _response_tokens(row["response"])
        ttft_ms = float(row["latency_ms"])
        tpot_ms = ttft_ms / tokens
        correct = bool(row["correct"])
        passes = (
            (correct or not slo.require_correct)
            and ttft_ms <= slo.max_ttft_ms
            and tpot_ms <= slo.max_tpot_ms
        )
        scores.append(
            CandidateScore(
                ckpt=row["ckpt"],
                correct=correct,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                estimated_cost=_mock_cost(row["size_mb"], row["latency_ms"]),
                passes=passes,
            )
        )
    return scores


def effective_goodput(scores: Iterable[CandidateScore], request_rate_rps: float) -> Dict[str, float]:
    items = list(scores)
    if not items:
        return {"attainment": 0.0, "goodput_rps": 0.0, "cost_per_good_request": float("inf")}
    passed = [s for s in items if s.passes]
    attainment = len(passed) / len(items)
    total_cost = sum(s.estimated_cost for s in items)
    return {
        "attainment": round(attainment, 4),
        "goodput_rps": round(request_rate_rps * attainment, 4),
        "cost_per_good_request": round(total_cost / max(1, len(passed)), 6),
    }


def rank_candidates(scores: Iterable[CandidateScore]) -> List[CandidateScore]:
    """Passing models first, then cheaper, then lower TTFT."""
    return sorted(scores, key=lambda s: (not s.passes, s.estimated_cost, s.ttft_ms))
