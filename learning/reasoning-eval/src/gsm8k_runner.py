"""GSM8K (Cobbe 2021) — grade school math 8.5k.

Real GSM8K has 7.5k train + 1k test. Each problem has multi-step natural
language solution ending with `#### N` exact answer.
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, ReasoningResult, accuracy, extract_gsm8k, make_dummy_model, make_mock_model, numeric_equal


_MICRO_GSM8K: List[Dict] = [
    {"qid": "gsm_1",
     "q": "Janet has 16 eggs. She eats 3 for breakfast and gives away 4. "
          "She sells the rest at $2 each. How much does she make?",
     "gold": "18"},
    {"qid": "gsm_2",
     "q": "Sam has $50. He buys 3 books at $7 each and 2 pens at $3 each. "
          "How much money does he have left?",
     "gold": "23"},
    {"qid": "gsm_3",
     "q": "A train travels at 60mph. How far does it travel in 2.5 hours?",
     "gold": "150"},
    {"qid": "gsm_4",
     "q": "A bakery sells 12 cakes/day at $25 each. What is the weekly revenue?",
     "gold": "2100"},
    {"qid": "gsm_5",
     "q": "Mary has 24 apples. She gives 1/3 to John, 1/4 of remainder to Sue. "
          "How many does she have left?",
     "gold": "12"},
    {"qid": "gsm_6",
     "q": "A class has 30 students, 60% are girls. How many boys?",
     "gold": "12"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _MICRO_GSM8K:
        prompt = (f"[qid={r['qid']}]\n"
                  f"Question: {r['q']}\n"
                  f"Let's think step by step. Final line should be '#### N'.\n"
                  f"Answer:")
        out.append({"qid": r["qid"], "prompt": prompt, "gold": r["gold"]})
    return out


def run_gsm8k(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 256)
        pred = extract_gsm8k(text) or ""
        rs.append(ReasoningResult(
            qid=d["qid"], pred=pred, gold=d["gold"],
            correct=numeric_equal(pred, d["gold"]),
        ))
    return rs


def _self_test() -> int:
    # dummy answers all 0 → 0% accuracy
    rs = run_gsm8k(make_dummy_model("0"))
    assert accuracy(rs) == 0.0
    # oracle
    gold_map = {r["qid"]: f"#### {r['gold']}" for r in _MICRO_GSM8K}
    rs2 = run_gsm8k(make_mock_model(gold_map))
    assert accuracy(rs2) == 1.0
    return 0


def _demo() -> None:
    """Visible demo: run the real scorer on mock models, print live accuracy."""
    print(f"GSM8K micro-set: {len(_MICRO_GSM8K)} problems")
    # Baseline: a dummy model that always says 0 -> should score ~0.
    base = run_gsm8k(make_dummy_model("0"))
    # Oracle: a mock model wired to the gold answers -> should score 1.0.
    gold_map = {r["qid"]: f"Reason... #### {r['gold']}" for r in _MICRO_GSM8K}
    oracle = run_gsm8k(make_mock_model(gold_map))
    print(f"  dummy('0')   accuracy = {accuracy(base):.2f}  "
          f"(pred for {base[0].qid}: {base[0].pred!r} vs gold {base[0].gold!r})")
    print(f"  oracle       accuracy = {accuracy(oracle):.2f}  "
          f"(pred for {oracle[0].qid}: {oracle[0].pred!r} vs gold {oracle[0].gold!r})")
    print("  -> accuracy is computed live: extract '#### N' -> numeric_equal vs gold.")


if __name__ == "__main__":
    f = _self_test()
    print(f"gsm8k_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
