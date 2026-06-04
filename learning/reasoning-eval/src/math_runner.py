"""MATH (Hendrycks 2021) — competition math.

12500 problems × 5 difficulty levels × 7 subjects (algebra/calculus/geometry/...).
Standard answer format: `\\boxed{...}`.
"""
from __future__ import annotations

from typing import Dict, List

from common import (
    ModelFn,
    ReasoningResult,
    accuracy,
    extract_boxed,
    extract_gsm8k,
    make_dummy_model,
    make_mock_model,
    numeric_equal,
)


_MICRO_MATH: List[Dict] = [
    {"qid": "m_1", "level": 1, "subject": "algebra",
     "q": "Solve for x: 2(x + 3) = 14",
     "gold": "4"},
    {"qid": "m_2", "level": 2, "subject": "geometry",
     "q": "What is the area of a triangle with base 6 and height 8?",
     "gold": "24"},
    {"qid": "m_3", "level": 3, "subject": "algebra",
     "q": "Find x: x^2 - 5x + 6 = 0 (larger root)",
     "gold": "3"},
    {"qid": "m_4", "level": 4, "subject": "calculus",
     "q": "Compute d/dx of x^3 + 2x at x=1",
     "gold": "5"},
    {"qid": "m_5", "level": 5, "subject": "number_theory",
     "q": "How many positive divisors does 36 have?",
     "gold": "9"},
    {"qid": "m_6", "level": 5, "subject": "combinatorics",
     "q": "In how many ways can 4 people sit in 4 chairs?",
     "gold": "24"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _MICRO_MATH:
        prompt = (
            f"[qid={r['qid']}] (Level {r['level']}, {r['subject']})\n"
            f"Problem: {r['q']}\n"
            f"Show your work. Put the final answer in \\boxed{{}}.\n"
            f"Solution:"
        )
        out.append({"qid": r["qid"], "prompt": prompt, "gold": r["gold"],
                    "meta": {"level": r["level"], "subject": r["subject"]}})
    return out


def run_math(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 256)
        pred = extract_boxed(text) or extract_gsm8k(text) or ""
        rs.append(ReasoningResult(
            qid=d["qid"], pred=pred, gold=d["gold"],
            correct=numeric_equal(pred, d["gold"]),
            meta=d["meta"],
        ))
    return rs


def by_level(rs: List[ReasoningResult]) -> Dict[int, float]:
    buckets: Dict[int, List[ReasoningResult]] = {}
    for r in rs:
        buckets.setdefault(r.meta["level"], []).append(r)
    return {k: accuracy(v) for k, v in buckets.items()}


def _self_test() -> int:
    rs = run_math(make_dummy_model("0"))
    assert accuracy(rs) == 0.0
    gold = {r["qid"]: f"\\boxed{{{r['gold']}}}" for r in _MICRO_MATH}
    rs2 = run_math(make_mock_model(gold))
    assert accuracy(rs2) == 1.0
    levels = by_level(rs2)
    assert set(levels.keys()) >= {1, 2, 3, 4, 5}
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"math_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
