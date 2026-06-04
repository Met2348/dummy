"""Humanity's Last Exam (Scale AI, 2025) — mock.

Real HLE: 3000 questions by ~800 PhD experts. We ship 5 mock samples
that demonstrate the format. Top models (Claude 3.7, o3) score 5-15%.
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, ReasoningResult, accuracy, make_dummy_model, make_mock_model


_MICRO_HLE: List[Dict] = [
    {"qid": "hle_1", "domain": "linguistics",
     "q": "What is the proto-form of Hungarian 'víz' (water) in "
          "Proto-Finno-Ugric?",
     "gold": "wete"},
    {"qid": "hle_2", "domain": "mathematics",
     "q": "What is the genus of the modular curve X_0(11)?",
     "gold": "1"},
    {"qid": "hle_3", "domain": "biology",
     "q": "Which RNA editing enzyme converts adenosine to inosine?",
     "gold": "ADAR"},
    {"qid": "hle_4", "domain": "history",
     "q": "In which year did the Hittite king Mursili I sack Babylon?",
     "gold": "1595"},
    {"qid": "hle_5", "domain": "chemistry",
     "q": "What is the molecular geometry of XeF4?",
     "gold": "square planar"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _MICRO_HLE:
        prompt = (f"[qid={r['qid']}] (Domain: {r['domain']})\n"
                  f"Question: {r['q']}\n"
                  f"Provide a short answer.\n"
                  f"Answer:")
        out.append({"qid": r["qid"], "prompt": prompt, "gold": r["gold"],
                    "meta": {"domain": r["domain"]}})
    return out


def run_hle(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 64).strip().lower()
        gold = d["gold"].lower()
        rs.append(ReasoningResult(
            qid=d["qid"], pred=text, gold=d["gold"],
            correct=(gold in text),
            meta=d["meta"],
        ))
    return rs


def _self_test() -> int:
    rs = run_hle(make_dummy_model("foo"))
    assert accuracy(rs) == 0.0
    gold = {r["qid"]: r["gold"] for r in _MICRO_HLE}
    rs2 = run_hle(make_mock_model(gold))
    assert accuracy(rs2) == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"hle_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
