"""AIME 2024/2025 — American Invitational Mathematics Examination.

30 questions, integer answer 0-999. The flagship benchmark for o1/R1
class models — Sonnet 3.7 ~54%, o1 ~83%, R1 ~80%.
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, ReasoningResult, accuracy, extract_boxed, make_dummy_model, make_mock_model, pass_at_k


_MICRO_AIME: List[Dict] = [
    {"qid": "aime_1", "year": 2024,
     "q": "Let N be the largest 3-digit number divisible by 7. What is N?",
     "gold": "994"},
    {"qid": "aime_2", "year": 2024,
     "q": "Find the smallest positive integer n such that n^2 > 1000.",
     "gold": "32"},
    {"qid": "aime_3", "year": 2025,
     "q": "How many positive divisors does 720 have?",
     "gold": "30"},
    {"qid": "aime_4", "year": 2025,
     "q": "Sum of integers from 1 to 100.",
     "gold": "5050"},
    {"qid": "aime_5", "year": 2025,
     "q": "Number of ways to arrange the letters in MISSISSIPPI.",
     "gold": "34650"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _MICRO_AIME:
        prompt = (
            f"[qid={r['qid']}] (AIME {r['year']})\n"
            f"Problem: {r['q']}\n"
            f"Answer must be an integer 0-999. Put it in \\boxed{{}}.\n"
            f"Solution:"
        )
        out.append({"qid": r["qid"], "prompt": prompt, "gold": r["gold"],
                    "meta": {"year": r["year"]}})
    return out


def run_aime(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 512)
        pred = (extract_boxed(text) or "").strip()
        rs.append(ReasoningResult(
            qid=d["qid"], pred=pred, gold=d["gold"],
            correct=(pred == d["gold"]),
            meta=d["meta"],
        ))
    return rs


def run_aime_passk(model: ModelFn, k: int = 4) -> Dict[int, float]:
    """Run k independent samples per question, compute pass@1/k.

    Real AIME evaluation uses pass@k because greedy is brittle on
    competition problems. Here mock model is deterministic, so all k
    samples produce the same answer.
    """
    per_q = []
    for d in build_prompts():
        sols = []
        for _ in range(k):
            text = model(d["prompt"], 512)
            pred = (extract_boxed(text) or "").strip()
            sols.append(pred == d["gold"])
        per_q.append(sols)
    return pass_at_k(per_q)


def _self_test() -> int:
    rs = run_aime(make_dummy_model("0"))
    assert accuracy(rs) == 0.0
    gold = {r["qid"]: f"\\boxed{{{r['gold']}}}" for r in _MICRO_AIME}
    oracle = make_mock_model(gold)
    rs2 = run_aime(oracle)
    assert accuracy(rs2) == 1.0
    pk = run_aime_passk(oracle, k=4)
    assert pk.get(1, 0.0) == 1.0
    assert pk.get(4, 0.0) == 1.0
    return 0


def _demo() -> None:
    """Visible demo: run the real scorer + pass@k on mock models, print live accuracy."""
    print(f"AIME micro-set: {len(_MICRO_AIME)} problems (integer answers 0-999)")
    base = run_aime(make_dummy_model("0"))
    gold = {r["qid"]: f"...\\boxed{{{r['gold']}}}" for r in _MICRO_AIME}
    oracle = make_mock_model(gold)
    rs = run_aime(oracle)
    print(f"  dummy('0')   accuracy = {accuracy(base):.2f}")
    print(f"  oracle       accuracy = {accuracy(rs):.2f}")
    pk = run_aime_passk(oracle, k=4)
    print(f"  oracle pass@k = {{ {', '.join(f'{k}:{v:.2f}' for k, v in sorted(pk.items()))} }}  "
          f"(deterministic mock -> all k samples agree)")
    print("  -> accuracy is computed live: extract \\boxed{} -> exact match vs gold.")


if __name__ == "__main__":
    f = _self_test()
    print(f"aime_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
