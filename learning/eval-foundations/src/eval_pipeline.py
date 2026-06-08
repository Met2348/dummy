"""Capstone - 4-bench joint runner producing a markdown report.

Combines: MMLU + BBH + TruthfulQA + commonsense, all on the same model.
Output: per-bench accuracy + overall summary + ASCII bar chart.
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, accuracy, make_mock_model, make_random_model
import mmlu_runner
import bbh_runner
import truthfulqa_runner
import commonsense_runner


def run_all_benches(model: ModelFn) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["mmlu"] = accuracy(mmlu_runner.run_mmlu(model))
    out["bbh"] = accuracy(bbh_runner.run_bbh(model))
    out["truthfulqa"] = accuracy(truthfulqa_runner.run_truthfulqa(model))
    out["commonsense"] = accuracy(commonsense_runner.run_commonsense(model))
    return out


def ascii_bar(value: float, width: int = 20) -> str:
    n = int(round(value * width))
    return "[" + "#" * n + " " * (width - n) + f"] {value*100:5.1f}%"


def to_md(results: Dict[str, float], model_label: str = "test_model") -> str:
    avg = sum(results.values()) / max(1, len(results))
    lines = [
        f"# Eval pipeline report - `{model_label}`",
        "",
        "| benchmark | accuracy | bar |",
        "|---|---:|---|",
    ]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.3f} | `{ascii_bar(v)}` |")
    lines.extend([
        "",
        f"**Overall average:** {avg:.3f} ({avg*100:.1f}%)",
        "",
        "## Notes",
        "- mmlu/bbh/commonsense: 4-option chance ~ 25%",
        "- mmlu_pro: 10-option chance ~ 10%",
        "- truthfulqa: anti-imitation bias, higher acc != truer; check error patterns",
    ])
    return "\n".join(lines)


def _self_test() -> int:
    rand = make_random_model(seed=0)
    res = run_all_benches(rand)
    assert set(res.keys()) == {"mmlu", "bbh", "truthfulqa", "commonsense"}
    for v in res.values():
        assert 0.0 <= v <= 1.0
    md = to_md(res, "random_baseline")
    assert "Overall average" in md
    # ASCII bar
    assert ascii_bar(0.5, 10).startswith("[#####")
    assert ascii_bar(0.0).startswith("[ ")
    assert ascii_bar(1.0).startswith("[###############")
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"eval_pipeline.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    rand = make_random_model(seed=0)
    print(to_md(run_all_benches(rand), "random_baseline"))
