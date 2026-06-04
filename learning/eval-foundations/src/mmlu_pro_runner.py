"""MMLU-Pro (Wang et al., 2024) — 10-option MCQ, harder than MMLU.

Key changes vs original MMLU:
1. 10 options A-J (chance level 10% not 25%)
2. Harder questions (reasoning-intensive)
3. Decontaminated from training corpora
"""
from __future__ import annotations

from typing import Dict, List

from common import (
    EvalResult,
    EvalSample,
    ModelFn,
    accuracy,
    extract_letter,
    make_mock_model,
    make_random_model,
)


_MICRO_MMLU_PRO: List[Dict] = [
    {"qid": "pro_1", "category": "math",
     "q": "If f(x) = x^3 - 3x + 1, what is f(2)?",
     "options": ["1", "3", "5", "7", "9", "11", "13", "15", "17", "19"],
     "gold": "B"},  # f(2) = 8 - 6 + 1 = 3
    {"qid": "pro_2", "category": "physics",
     "q": "A ball is dropped from 20m. Approx fall time (g=10)?",
     "options": ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s", "3.5s", "4s", "4.5s", "5s"],
     "gold": "D"},  # t = sqrt(2*20/10) = 2s
    {"qid": "pro_3", "category": "law",
     "q": "Habeas corpus protects against:",
     "options": ["Tax fraud", "Unlawful detention", "Copyright", "Slander",
                 "Trespass", "Negligence", "Defamation", "Breach", "Theft", "Murder"],
     "gold": "B"},
    {"qid": "pro_4", "category": "chemistry",
     "q": "Atomic number of carbon:",
     "options": ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13"],
     "gold": "C"},
    {"qid": "pro_5", "category": "economics",
     "q": "Comparative advantage was formulated by:",
     "options": ["Smith", "Ricardo", "Marx", "Keynes", "Friedman",
                 "Mill", "Malthus", "Mises", "Hayek", "Samuelson"],
     "gold": "B"},
]


def build_samples() -> List[EvalSample]:
    return [
        EvalSample(
            qid=r["qid"],
            prompt=_format_10opt(r),
            gold=r["gold"],
            meta={"category": r["category"]},
        )
        for r in _MICRO_MMLU_PRO
    ]


def _format_10opt(row: Dict) -> str:
    body = [f"[qid={row['qid']}] (Category: {row['category']})",
            f"Question: {row['q']}"]
    for i, opt in enumerate(row["options"]):
        body.append(f"{chr(ord('A')+i)}. {opt}")
    body.append("Answer:")
    return "\n".join(body)


def run_mmlu_pro(model: ModelFn) -> List[EvalResult]:
    samples = build_samples()
    out = []
    for s in samples:
        text = model(s.prompt, max_new_tokens=4)
        # Pro has A-J, extend letter extraction
        pred = ""
        for ch in text.strip():
            if ch in "ABCDEFGHIJ":
                pred = ch
                break
        if not pred:
            pred = extract_letter(text) or ""
        out.append(EvalResult(
            qid=s.qid, pred=pred, gold=s.gold,
            correct=(pred == s.gold), meta=s.meta,
        ))
    return out


def _self_test() -> int:
    samples = build_samples()
    assert len(samples) == 5
    assert all(len(_MICRO_MMLU_PRO[i]["options"]) == 10 for i in range(5))
    # Random chance ≈ 10% (5 samples too few to assert)
    rand = make_random_model(seed=0, choices="ABCDEFGHIJ")
    rs = run_mmlu_pro(rand)
    assert 0.0 <= accuracy(rs) <= 1.0
    # Oracle
    gold = {s.qid: s.gold for s in samples}
    oracle = make_mock_model(gold)
    rs2 = run_mmlu_pro(oracle)
    assert accuracy(rs2) == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"mmlu_pro_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
