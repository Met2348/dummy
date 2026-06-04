"""Classic commonsense benchmarks — HellaSwag / ARC / Winogrande (toy).

These were the workhorses 2019-2023 but have largely saturated.
We ship 2 q each (6 total) to demonstrate the format.
"""
from __future__ import annotations

from typing import Dict, List

from common import EvalResult, EvalSample, ModelFn, accuracy, group_accuracy, make_mock_model, make_random_model


_DATA: List[Dict] = [
    # HellaSwag (sentence completion)
    {"qid": "hs_1", "bench": "hellaswag",
     "q": "A man is shoveling snow off his driveway. He...",
     "options": ["puts the shovel into space",
                 "continues to clear the path while wearing a winter coat",
                 "drinks the snow", "flies away"],
     "gold": "B"},
    {"qid": "hs_2", "bench": "hellaswag",
     "q": "A chef is dicing onions. She...",
     "options": ["eats them raw", "places them in a pan to cook",
                 "throws them at the customer", "fills a bathtub with them"],
     "gold": "B"},
    # ARC (science MCQ)
    {"qid": "arc_1", "bench": "arc",
     "q": "Which of the following is a renewable energy source?",
     "options": ["Coal", "Solar", "Natural gas", "Petroleum"],
     "gold": "B"},
    {"qid": "arc_2", "bench": "arc",
     "q": "Plants make food through:",
     "options": ["Respiration", "Photosynthesis", "Digestion", "Excretion"],
     "gold": "B"},
    # Winogrande (pronoun resolution)
    {"qid": "wg_1", "bench": "winogrande",
     "q": "The trophy didn't fit in the suitcase because IT was too big. IT refers to:",
     "options": ["trophy", "suitcase", "neither", "both"],
     "gold": "A"},
    {"qid": "wg_2", "bench": "winogrande",
     "q": "The trophy didn't fit in the suitcase because IT was too small. IT refers to:",
     "options": ["trophy", "suitcase", "neither", "both"],
     "gold": "B"},
]


def build_samples() -> List[EvalSample]:
    samples = []
    for r in _DATA:
        body = [f"[qid={r['qid']}] (Bench: {r['bench']})", f"Q: {r['q']}"]
        for i, opt in enumerate(r["options"]):
            body.append(f"{chr(ord('A')+i)}. {opt}")
        body.append("A:")
        samples.append(EvalSample(
            qid=r["qid"], prompt="\n".join(body), gold=r["gold"],
            meta={"bench": r["bench"]},
        ))
    return samples


def run_commonsense(model: ModelFn) -> List[EvalResult]:
    samples = build_samples()
    out = []
    for s in samples:
        text = model(s.prompt, max_new_tokens=4).strip()
        pred = ""
        for ch in text:
            if ch in "ABCD":
                pred = ch
                break
        out.append(EvalResult(
            qid=s.qid, pred=pred, gold=s.gold,
            correct=(pred == s.gold), meta=s.meta,
        ))
    return out


def summarize(rs: List[EvalResult]) -> Dict:
    return {"accuracy": accuracy(rs), "by_bench": group_accuracy(rs, by="bench")}


def _self_test() -> int:
    samples = build_samples()
    assert len(samples) == 6
    gold = {s.qid: s.gold for s in samples}
    oracle = make_mock_model(gold)
    rs2 = run_commonsense(oracle)
    assert accuracy(rs2) == 1.0
    summ = summarize(rs2)
    assert len(summ["by_bench"]) == 3
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"commonsense_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
