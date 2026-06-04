"""Capstone — Reasoning bench comparison: 2 mock models × 5 benches.

Models:
- "gpt2_base": dummy model returning constant 0 (simulating untrained baseline)
- "r1_tiny": mock model returning correct answers (simulating R1-Zero ckpt)

Benches:
- GSM8K, MATH, AIME, GPQA, ZebraLogic

Output: 2 × 5 = 10-cell comparison table.
"""
from __future__ import annotations

from typing import Dict

from common import ModelFn, accuracy, make_dummy_model, make_mock_model
import gsm8k_runner
import math_runner
import aime_runner
import gpqa_runner
import zebra_logic


# === The "two ckpts" ===

def make_gpt2_baseline() -> ModelFn:
    """Untrained — always answers 0."""
    return make_dummy_model("0")


def make_r1_tiny() -> ModelFn:
    """Mock R1-tiny that answers everything correctly via gold map."""
    gold = {}
    # GSM8K
    for r in gsm8k_runner._MICRO_GSM8K:
        gold[r["qid"]] = f"Let's solve step by step. #### {r['gold']}"
    # MATH
    for r in math_runner._MICRO_MATH:
        gold[r["qid"]] = f"...\\boxed{{{r['gold']}}}"
    # AIME
    for r in aime_runner._MICRO_AIME:
        gold[r["qid"]] = f"...\\boxed{{{r['gold']}}}"
    # GPQA
    for r in gpqa_runner._MICRO_GPQA:
        gold[r["qid"]] = f"...Final: {r['gold']}"
    # Zebra
    for r in zebra_logic._PUZZLES:
        gold[r["qid"]] = f"...{r['gold']}"
    return make_mock_model(gold)


def run_all(model: ModelFn) -> Dict[str, float]:
    return {
        "gsm8k": accuracy(gsm8k_runner.run_gsm8k(model)),
        "math": accuracy(math_runner.run_math(model)),
        "aime": accuracy(aime_runner.run_aime(model)),
        "gpqa": accuracy(gpqa_runner.run_gpqa(model)),
        "zebra": accuracy(zebra_logic.run_zebra(model)),
    }


def to_md(rows: Dict[str, Dict[str, float]]) -> str:
    benches = ["gsm8k", "math", "aime", "gpqa", "zebra"]
    head = "| ckpt | " + " | ".join(benches) + " | avg |"
    sep = "|---|" + "---|" * (len(benches) + 1)
    lines = [head, sep]
    for name, r in rows.items():
        vals = [f"{r[b]:.2f}" for b in benches]
        avg = sum(r[b] for b in benches) / len(benches)
        lines.append(f"| {name} | " + " | ".join(vals) + f" | {avg:.2f} |")
    return "\n".join(lines)


def run_capstone() -> Dict:
    base = run_all(make_gpt2_baseline())
    r1 = run_all(make_r1_tiny())
    rows = {"gpt2_base (vanilla)": base, "r1_tiny (RL-trained)": r1}
    return {"table": to_md(rows), "rows": rows}


def _self_test() -> int:
    out = run_capstone()
    base = out["rows"]["gpt2_base (vanilla)"]
    r1 = out["rows"]["r1_tiny (RL-trained)"]
    # Baseline weak across most benches
    assert sum(base.values()) < 1.0
    # R1-tiny aces all benches
    assert all(v == 1.0 for v in r1.values())
    # Table contains both names + columns
    table = out["table"]
    assert "gsm8k" in table and "gpt2_base" in table and "r1_tiny" in table
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"capstone_reasoning_compare.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    out = run_capstone()
    print(out["table"])
