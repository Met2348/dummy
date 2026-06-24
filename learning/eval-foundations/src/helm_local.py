"""HELM (Stanford 2022) — 简化本地复现.

Real HELM has 16 scenarios × 7 metrics. Here we ship a teaching micro-HELM
with 4 scenarios and 4 metrics — enough to show the holistic framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from common import EvalResult, ModelFn, accuracy, make_mock_model, make_random_model


# === 4 scenarios ===

SCENARIOS = ["knowledge", "reasoning", "summarization", "robustness"]


@dataclass
class HelmCell:
    scenario: str
    metric: str
    value: float


# === Toy mini benchmarks ===

_KNOWLEDGE_QA = [
    {"qid": "k1", "q": "[qid=k1] Capital of France?", "gold": "Paris"},
    {"qid": "k2", "q": "[qid=k2] Speed of light (km/s)?", "gold": "300000"},
    {"qid": "k3", "q": "[qid=k3] Year humans landed on the moon?", "gold": "1969"},
]

_REASONING_QA = [
    {"qid": "r1", "q": "[qid=r1] If x+5=12, what is x?", "gold": "7"},
    {"qid": "r2", "q": "[qid=r2] All A are B. All B are C. Are all A C? (yes/no)", "gold": "yes"},
]

_SUMM_DOC = [
    {"qid": "s1",
     "q": "[qid=s1] Summarize: The quick brown fox jumps over the lazy dog.",
     "ref": "Fox jumps over dog"},
]

_ROBUST_QA = [
    # Same as knowledge but with typos / case changes
    {"qid": "rb1", "q": "[qid=rb1] CaPiTal of FrAnCe?", "gold": "Paris"},
    {"qid": "rb2", "q": "[qid=rb2] Speed of liiight (km/s)?", "gold": "300000"},
]


def _exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def _rouge1_proxy(pred: str, ref: str) -> float:
    """Crude unigram overlap, no stopword removal — enough for teaching."""
    pset = set(pred.lower().split())
    rset = set(ref.lower().split())
    if not rset:
        return 0.0
    return len(pset & rset) / len(rset)


# === Per-scenario eval ===

def _eval_qa(qs: List[Dict], model: ModelFn) -> float:
    rs = []
    for q in qs:
        pred = model(q["q"], max_new_tokens=16)
        rs.append(EvalResult(q["qid"], pred, q["gold"],
                             correct=_exact_match(pred, q["gold"])))
    return accuracy(rs)


def _eval_summ(docs: List[Dict], model: ModelFn) -> float:
    scores = []
    for d in docs:
        pred = model(d["q"], max_new_tokens=32)
        scores.append(_rouge1_proxy(pred, d["ref"]))
    return sum(scores) / len(scores) if scores else 0.0


def run_helm_local(model: ModelFn) -> List[HelmCell]:
    cells = [
        HelmCell("knowledge", "exact_match", _eval_qa(_KNOWLEDGE_QA, model)),
        HelmCell("reasoning", "exact_match", _eval_qa(_REASONING_QA, model)),
        HelmCell("summarization", "rouge1_proxy", _eval_summ(_SUMM_DOC, model)),
        HelmCell("robustness", "exact_match", _eval_qa(_ROBUST_QA, model)),
    ]
    return cells


def render_table(cells: List[HelmCell]) -> str:
    lines = ["| scenario | metric | value |", "|---|---|---|"]
    for c in cells:
        lines.append(f"| {c.scenario} | {c.metric} | {c.value:.3f} |")
    return "\n".join(lines)


def _self_test() -> int:
    rand = make_random_model(seed=0)
    cells = run_helm_local(rand)
    assert len(cells) == 4
    assert {c.scenario for c in cells} == set(SCENARIOS)
    # Oracle model for QA portions
    oracle_map = {
        "k1": "Paris", "k2": "300000", "k3": "1969",
        "r1": "7", "r2": "yes",
        "s1": "Fox jumps over dog",
        "rb1": "Paris", "rb2": "300000",
    }
    oracle = make_mock_model(oracle_map)
    cells2 = run_helm_local(oracle)
    by_scen = {c.scenario: c for c in cells2}
    assert by_scen["knowledge"].value == 1.0
    assert by_scen["reasoning"].value == 1.0
    assert by_scen["summarization"].value > 0.5
    assert by_scen["robustness"].value == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"helm_local.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    # Demo: render the micro-HELM scenario x metric table for a random model
    rand = make_random_model(seed=42)
    print(render_table(run_helm_local(rand)))
