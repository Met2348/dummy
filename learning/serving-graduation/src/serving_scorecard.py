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


def demo() -> None:
    print("=== 毕业服务评分卡（correctness 门 + TTFT/TPOT SLO + goodput）===")
    report = {"results": [
        {"ckpt": "0.5B", "response": "<think>brief</think><answer>42</answer>",
         "latency_ms": 120, "correct": True, "size_mb": 500},
        {"ckpt": "7B", "response": "<think>some reasoning steps</think><answer>42</answer>",
         "latency_ms": 450, "correct": True, "size_mb": 7000},
        {"ckpt": "70B-wrong", "response": "<think>...</think><answer>41</answer>",
         "latency_ms": 2200, "correct": False, "size_mb": 70000},
    ]}
    slo = GraduationSLO(max_ttft_ms=900, max_tpot_ms=80)
    scores = score_report(report, slo)
    for s in scores:
        print(f"  {s.ckpt:>9}: correct={s.correct!s:5} ttft={s.ttft_ms:>6.0f}ms "
              f"tpot={s.tpot_ms:>5.1f}ms cost={s.estimated_cost:.3f} passes={s.passes}")
    gp = effective_goodput(scores, request_rate_rps=10.0)
    print(f"goodput@10rps: attainment={gp['attainment']} good_rps={gp['goodput_rps']} "
          f"cost/good={gp['cost_per_good_request']}")
    print(f"排名(过 SLO 优先→便宜→低 TTFT): {[s.ckpt for s in rank_candidates(scores)]}")


if __name__ == "__main__":
    demo()
