"""Safety eval — measure classifier precision/recall on test set."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from common import (
    BENIGN_QUERIES,
    HARMFUL_QUERIES,
    INJECTION_QUERIES,
    GuardVerdict,
    confusion_matrix,
    precision_recall,
)


GuardFn = Callable[[str], GuardVerdict]


def evaluate_guard(guard: GuardFn) -> Dict:
    """Evaluate a guard on benign + harmful + injection queries.

    Goals:
      - benign → safe (avoid false positive)
      - harmful → unsafe (true positive)
      - injection → unsafe (true positive)
    """
    pairs: List[Tuple[str, str]] = []
    samples: List[Tuple[str, str]] = []
    for q in BENIGN_QUERIES:
        samples.append((q, "safe"))
    for q in HARMFUL_QUERIES + INJECTION_QUERIES:
        samples.append((q, "unsafe"))
    for q, true in samples:
        v = guard(q)
        pairs.append((true, v.label))
    cm = confusion_matrix(pairs)
    p, r, f1 = precision_recall(cm)
    return {
        "confusion": cm,
        "precision": p, "recall": r, "f1": f1,
        "n_samples": len(samples),
    }


def to_md(name: str, metrics: Dict) -> str:
    cm = metrics["confusion"]
    return "\n".join([
        f"# Safety guard report — `{name}`",
        f"- TP: {cm['tp']} (correctly flagged unsafe)",
        f"- FP: {cm['fp']} (benign falsely flagged)",
        f"- TN: {cm['tn']} (benign passed)",
        f"- FN: {cm['fn']} (unsafe missed)",
        f"- precision: {metrics['precision']:.2f}",
        f"- recall:    {metrics['recall']:.2f}",
        f"- F1:        {metrics['f1']:.2f}",
    ])


def _self_test() -> int:
    from llama_guard_mock import classify_input
    m = evaluate_guard(classify_input)
    assert m["n_samples"] == len(BENIGN_QUERIES) + len(HARMFUL_QUERIES) + len(INJECTION_QUERIES)
    # llama-guard mock catches some harmful but not all injection
    assert m["recall"] > 0
    md = to_md("llama_guard_mock", m)
    assert "F1" in md
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"safety_eval_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
