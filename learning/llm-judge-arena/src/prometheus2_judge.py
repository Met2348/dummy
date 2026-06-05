"""Prometheus 2 (2024) — open-source judge model mock.

Real Prometheus 2: 7B/8x7B model fine-tuned to score 1-5 against rubric.

Our toy: scores response by checking presence of rubric criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Rubric:
    name: str
    criteria: List[str]   # keywords/phrases each worth 1 point
    max_score: int = 5


COMMON_RUBRICS = {
    "helpfulness": Rubric(
        "helpfulness",
        ["step", "specifically", "example", "because", "therefore"],
    ),
    "factuality": Rubric(
        "factuality",
        ["according to", "in 19", "in 20", "is defined as", "the value is"],
    ),
    "clarity": Rubric(
        "clarity",
        ["first", "second", "to summarize", "in summary", "the key point is"],
    ),
}


def score_rubric(response: str, rubric: Rubric) -> int:
    hits = sum(1 for kw in rubric.criteria if kw.lower() in response.lower())
    # 1 (no hits) → 5 (>= len criteria hits)
    return max(1, min(rubric.max_score, 1 + hits))


def explain_score(response: str, rubric: Rubric) -> Dict:
    matched = [kw for kw in rubric.criteria if kw.lower() in response.lower()]
    return {
        "rubric": rubric.name,
        "score": score_rubric(response, rubric),
        "matched_criteria": matched,
        "missed_criteria": [kw for kw in rubric.criteria if kw not in matched],
    }


def _self_test() -> int:
    r = COMMON_RUBRICS["helpfulness"]
    assert score_rubric("", r) == 1
    assert score_rubric("step by step. Specifically, example here. Because therefore.",
                         r) == 5
    e = explain_score("step example", r)
    assert e["score"] == 3
    assert "step" in e["matched_criteria"]
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"prometheus2_judge.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
