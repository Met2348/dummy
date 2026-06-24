"""GPQA Diamond (Rein 2023) — Google-Proof Q&A.

448 PhD-level multiple choice in biology/physics/chemistry. Even with
google search humans get ~80%; closed-book chance level is 25%.
Top models (GPT-4o ~52%, Claude 3.7 ~68%).
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, ReasoningResult, accuracy, make_dummy_model, make_mock_model


_MICRO_GPQA: List[Dict] = [
    {"qid": "gpqa_1", "field": "physics",
     "q": "An electron is in 1s state of hydrogen. What is the most "
          "probable distance from nucleus (Bohr radius a_0)?",
     "options": ["0", "a_0", "2 a_0", "0.5 a_0"], "gold": "B"},
    {"qid": "gpqa_2", "field": "chemistry",
     "q": "Which is the strongest oxidizing agent at standard conditions?",
     "options": ["Cl2", "F2", "Br2", "I2"], "gold": "B"},
    {"qid": "gpqa_3", "field": "biology",
     "q": "In which cell organelle does aerobic respiration ATP production "
          "primarily occur?",
     "options": ["Cytosol", "Mitochondrion (matrix)",
                 "Mitochondrion (inner membrane)", "Chloroplast"],
     "gold": "C"},
    {"qid": "gpqa_4", "field": "physics",
     "q": "What is the dimension of action h?",
     "options": ["ML^2/T", "ML/T^2", "ML^2/T^2", "M/L"], "gold": "A"},
    {"qid": "gpqa_5", "field": "chemistry",
     "q": "The pKa of acetic acid is approximately:",
     "options": ["1.7", "4.7", "7.4", "12.0"], "gold": "B"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _MICRO_GPQA:
        body = [f"[qid={r['qid']}] (Field: {r['field']})",
                f"Question: {r['q']}"]
        for i, opt in enumerate(r["options"]):
            body.append(f"{chr(ord('A')+i)}. {opt}")
        body.append("Let's think step by step. End with 'Final: X'.")
        body.append("Solution:")
        out.append({"qid": r["qid"], "prompt": "\n".join(body),
                    "gold": r["gold"], "meta": {"field": r["field"]}})
    return out


def _extract(text: str) -> str:
    import re
    m = re.search(r"final[:\s]+([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else ""


def run_gpqa(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 256)
        pred = _extract(text)
        rs.append(ReasoningResult(
            qid=d["qid"], pred=pred, gold=d["gold"],
            correct=(pred == d["gold"]),
            meta=d["meta"],
        ))
    return rs


def _self_test() -> int:
    rs = run_gpqa(make_dummy_model("E"))  # no valid letter
    assert accuracy(rs) == 0.0
    gold = {r["qid"]: f"Final: {r['gold']}" for r in _MICRO_GPQA}
    rs2 = run_gpqa(make_mock_model(gold))
    assert accuracy(rs2) == 1.0
    return 0


def _demo() -> None:
    """Visible demo: run the real scorer on mock models, print live accuracy."""
    print(f"GPQA micro-set: {len(_MICRO_GPQA)} multiple-choice (chance = 25%)")
    base = run_gpqa(make_dummy_model("E"))  # 'E' is not a valid A-D letter
    gold = {r["qid"]: f"...Final: {r['gold']}" for r in _MICRO_GPQA}
    oracle = run_gpqa(make_mock_model(gold))
    print(f"  dummy('E')   accuracy = {accuracy(base):.2f}  (invalid letter -> 0)")
    print(f"  oracle       accuracy = {accuracy(oracle):.2f}  "
          f"(pred for {oracle[0].qid}: {oracle[0].pred!r} vs gold {oracle[0].gold!r})")
    print("  -> accuracy is computed live: extract 'Final: X' letter -> match vs gold.")


if __name__ == "__main__":
    f = _self_test()
    print(f"gpqa_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
