"""Big-Bench Hard (BBH, 2022) — 23 task collection of harder Big-Bench items.

Micro version with 3 tasks: tracking_shuffled_objects, date_understanding,
logical_deduction. 4 questions per task = 12 total.
"""
from __future__ import annotations

from typing import Dict, List

from common import EvalResult, EvalSample, ModelFn, accuracy, group_accuracy, make_mock_model, make_random_model


_TASKS: List[Dict] = [
    # tracking_shuffled_objects (3-object)
    {"qid": "track_1", "task": "tracking",
     "q": "Alice has a red ball, Bob has a blue ball, Carol has a green ball. "
          "Alice and Bob swap. Then Bob and Carol swap. Who has the red ball?",
     "options": ["Alice", "Bob", "Carol"], "gold": "C"},
    {"qid": "track_2", "task": "tracking",
     "q": "Apples: A=1, B=2, C=3. Swap A and C. Then swap B and A. B's count?",
     "options": ["1", "2", "3"], "gold": "C"},
    # date_understanding
    {"qid": "date_1", "task": "date",
     "q": "Today is March 5, 2024. What was the date 1 week ago?",
     "options": ["Feb 27, 2024", "Feb 28, 2024", "Mar 1, 2024"], "gold": "A"},
    {"qid": "date_2", "task": "date",
     "q": "Yesterday was 2024-02-29. What month was 3 months later?",
     "options": ["May", "June", "July"], "gold": "A"},
    # logical_deduction
    {"qid": "logic_1", "task": "logic",
     "q": "Tom is taller than Jerry. Jerry is taller than Sam. Who is shortest?",
     "options": ["Tom", "Jerry", "Sam"], "gold": "C"},
    {"qid": "logic_2", "task": "logic",
     "q": "If A->B and B->C, and A is true, what about C?",
     "options": ["True", "False", "Unknown"], "gold": "A"},
]


def build_samples() -> List[EvalSample]:
    samples = []
    for r in _TASKS:
        body = [f"[qid={r['qid']}] (Task: {r['task']})", f"Q: {r['q']}"]
        for i, opt in enumerate(r["options"]):
            body.append(f"{chr(ord('A')+i)}. {opt}")
        body.append("A:")
        samples.append(EvalSample(
            qid=r["qid"], prompt="\n".join(body), gold=r["gold"],
            meta={"task": r["task"]},
        ))
    return samples


def run_bbh(model: ModelFn) -> List[EvalResult]:
    samples = build_samples()
    out = []
    for s in samples:
        text = model(s.prompt, max_new_tokens=4).strip()
        pred = ""
        for ch in text:
            if ch in "ABC":
                pred = ch
                break
        out.append(EvalResult(
            qid=s.qid, pred=pred, gold=s.gold,
            correct=(pred == s.gold), meta=s.meta,
        ))
    return out


def summarize(rs: List[EvalResult]) -> Dict:
    return {"accuracy": accuracy(rs), "by_task": group_accuracy(rs, by="task")}


def _self_test() -> int:
    samples = build_samples()
    assert len(samples) == 6
    rand = make_random_model(seed=0, choices="ABC")
    rs = run_bbh(rand)
    assert 0.0 <= accuracy(rs) <= 1.0
    gold = {s.qid: s.gold for s in samples}
    oracle = make_mock_model(gold)
    rs2 = run_bbh(oracle)
    assert accuracy(rs2) == 1.0
    summ = summarize(rs2)
    assert len(summ["by_task"]) == 3
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"bbh_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
