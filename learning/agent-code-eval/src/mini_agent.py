"""Capstone — mini-agent running HumanEval + SWE-Bench + WebArena + BFCL.

Demonstrates the eval pattern for the "造 → 用 → 评" loop:
- Code: HumanEval / LiveCodeBench / SWE-Bench
- Web: WebArena
- Tool: BFCL

Output: 4 × 1 score table.
"""
from __future__ import annotations

import json
from typing import Dict

from common import ModelFn, make_mock_model, pass_rate
import humaneval_runner
import swebench_mock
import webarena_mock
import bfcl_runner
import livecodebench_mock


def run_all(model: ModelFn) -> Dict[str, float]:
    he = pass_rate(humaneval_runner.run_humaneval(model))
    lcb = pass_rate(livecodebench_mock.run_livecodebench(model))
    swe_results = swebench_mock.run_swebench_mock(model)
    swe = sum(1 for r in swe_results if r["passed"]) / max(1, len(swe_results))
    web_results = webarena_mock.run_webarena_mock(model)
    web = sum(1 for r in web_results if r["passed"]) / max(1, len(web_results))
    bfcl_results = bfcl_runner.run_bfcl(model)
    bf = sum(1 for r in bfcl_results if r["passed"]) / max(1, len(bfcl_results))
    return {
        "humaneval": he, "livecodebench": lcb,
        "swebench": swe, "webarena": web, "bfcl": bf,
    }


def to_md(scores: Dict[str, float], label: str = "model") -> str:
    lines = [f"# Mini-agent capstone — `{label}`",
             "| bench | score |", "|---|---:|"]
    for k, v in scores.items():
        lines.append(f"| {k} | {v:.2f} |")
    lines.append(f"| **avg** | **{sum(scores.values())/len(scores):.2f}** |")
    return "\n".join(lines)


def _self_test() -> int:
    # Empty → 0%
    empty = make_mock_model({}, default="")
    s_low = run_all(empty)
    assert all(v == 0.0 for v in s_low.values()), s_low
    # Build a "smart" model that knows all gold answers
    all_answers = {}
    # HumanEval refs
    for t in humaneval_runner._TASKS:
        all_answers[t["qid"]] = f"```python\n{t['ref']}\n```"
    # SWE
    all_answers["swe_1"] = f"```python\n{swebench_mock.REPO_FILE_FIXED}```"
    # Web
    all_answers["web_1"] = "go to item\nadd to cart\ngo to checkout\nconfirm order"
    # BFCL
    for t in bfcl_runner._TASKS:
        all_answers[t["qid"]] = json.dumps(t["expect"])
    # LCB refs
    for t in livecodebench_mock._TASKS:
        all_answers[t["qid"]] = f"```python\n{t['ref']}\n```"
    smart = make_mock_model(all_answers)
    s_hi = run_all(smart)
    assert all(v == 1.0 for v in s_hi.values()), s_hi
    md = to_md(s_hi, "smart")
    assert "avg" in md
    return 0


def _demo() -> None:
    """Visible demo: print the real 5-bench scorecards for empty vs smart models.

    Note: this capstone is a benchmark *aggregator* — each runner does a single
    generate -> exec/score pass over a mock model. It is not an iterative
    read->generate->run->revise agent loop; it shows how to assemble a 5-bench
    scorecard, with every cell computed by the real per-bench scorer.
    """
    empty = make_mock_model({}, default="")
    print(to_md(run_all(empty), "empty-model"))
    print()
    all_answers: Dict[str, str] = {}
    for t in humaneval_runner._TASKS:
        all_answers[t["qid"]] = f"```python\n{t['ref']}\n```"
    all_answers["swe_1"] = f"```python\n{swebench_mock.REPO_FILE_FIXED}```"
    all_answers["web_1"] = "go to item\nadd to cart\ngo to checkout\nconfirm order"
    for t in bfcl_runner._TASKS:
        all_answers[t["qid"]] = json.dumps(t["expect"])
    for t in livecodebench_mock._TASKS:
        all_answers[t["qid"]] = f"```python\n{t['ref']}\n```"
    print(to_md(run_all(make_mock_model(all_answers)), "smart-model"))
    print("\n-> every cell is a live pass_rate from its real per-bench scorer "
          "(empty=all 0.00, oracle=all 1.00).")


if __name__ == "__main__":
    f = _self_test()
    print(f"mini_agent.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
