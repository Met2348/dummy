"""Adapter to lm-evaluation-harness API style.

We don't actually depend on lm-eval (heavy install). This file shows the
official API surface and provides a duck-typed mock so users can swap in
the real library by editing 1 line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from common import ModelFn, EvalResult


@dataclass
class Task:
    """Mimics lm-eval's Task object."""
    name: str
    samples: List[Dict]   # each dict has {qid, prompt, gold}
    metric: str = "accuracy"


@dataclass
class TaskResult:
    name: str
    metric: str
    value: float
    n: int


def evaluate_tasks(model: ModelFn, tasks: List[Task]) -> List[TaskResult]:
    """Run model on each Task and aggregate by metric.

    The real lm-eval supports rich metrics (perplexity, BLEU, ROUGE,
    exact match, etc.); here we keep it to {'accuracy'} for teaching.
    """
    out = []
    for t in tasks:
        n_ok = 0
        for s in t.samples:
            pred = model(s["prompt"], max_new_tokens=4).strip()
            letter = ""
            for ch in pred:
                if ch in "ABCDEFGHIJ":
                    letter = ch
                    break
            if letter == s["gold"]:
                n_ok += 1
        acc = n_ok / max(1, len(t.samples))
        out.append(TaskResult(name=t.name, metric=t.metric, value=acc, n=len(t.samples)))
    return out


def render_results(results: List[TaskResult]) -> str:
    lines = ["| task | metric | n | value |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.metric} | {r.n} | {r.value:.3f} |")
    return "\n".join(lines)


def _self_test() -> int:
    from common import make_mock_model
    tasks = [
        Task(name="dummy_mc",
             samples=[
                 {"qid": "d1", "prompt": "[qid=d1] Pick: A. cat B. dog\nA:",
                  "gold": "A"},
                 {"qid": "d2", "prompt": "[qid=d2] Pick: A. cat B. dog\nA:",
                  "gold": "B"},
             ]),
    ]
    m = make_mock_model({"d1": "A", "d2": "B"})
    res = evaluate_tasks(m, tasks)
    assert len(res) == 1
    assert res[0].value == 1.0
    assert res[0].n == 2
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"lm_eval_adapter.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
