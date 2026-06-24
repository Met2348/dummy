"""Tool-augmented math — model calls a calculator.

Demonstrates the "model writes code, sandbox executes" pattern used by
Program-of-Thoughts (PoT) and Toolformer. Works on GSM8K-style problems
where the model emits a small program, we execute, then return the
result.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from common import ModelFn, ReasoningResult, accuracy, make_mock_model, numeric_equal


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.+?)```", re.DOTALL)


def _safe_eval(expr: str) -> Optional[str]:
    """Evaluate a tiny arithmetic expression."""
    allowed = set("0123456789+-*/.() ")
    if any(c not in allowed for c in expr):
        return None
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _safe_exec_block(code: str) -> Optional[str]:
    """Exec a small Python block, capture last variable 'answer'."""
    forbidden = ("import", "open(", "__", "exec(", "eval(", "compile(", "os.", "sys.")
    if any(f in code for f in forbidden):
        return None
    ns: Dict = {"__builtins__": {"range": range, "sum": sum, "len": len,
                                  "abs": abs, "min": min, "max": max,
                                  "pow": pow, "round": round, "int": int,
                                  "float": float, "all": all, "any": any,
                                  "sorted": sorted, "enumerate": enumerate,
                                  "list": list, "set": set, "dict": dict}}
    try:
        exec(code, ns)
        if "answer" in ns:
            return str(ns["answer"])
    except Exception:
        return None
    return None


def run_with_tool(model: ModelFn, question: str, qid: str = "tq") -> str:
    """Run model, extract code, execute, return numeric answer."""
    prompt = (f"[qid={qid}]\n"
              f"Question: {question}\n"
              f"Write a small Python block that sets variable `answer` "
              f"to the numeric result.\n"
              f"```python\n# your code\nanswer = ?\n```")
    text = model(prompt, 256)
    m = _CODE_BLOCK_RE.search(text)
    if not m:
        return ""
    return _safe_exec_block(m.group(1).strip()) or ""


# === Test scenarios ===

_DEMOS: List[Dict] = [
    {"qid": "ta_1", "q": "What is 17 * 23?", "gold": "391"},
    {"qid": "ta_2", "q": "Sum of integers 1 to 100?", "gold": "5050"},
    {"qid": "ta_3", "q": "How many primes < 20?", "gold": "8"},
]


def _self_test() -> int:
    # Build a mock model that returns code blocks for each qid
    answers = {
        "ta_1": "```python\nanswer = 17 * 23\n```",
        "ta_2": "```python\nanswer = sum(range(1, 101))\n```",
        "ta_3": "```python\nprimes = [n for n in range(2, 20) if all(n % d != 0 for d in range(2, n))]\nanswer = len(primes)\n```",
    }
    m = make_mock_model(answers)
    rs = []
    for d in _DEMOS:
        pred = run_with_tool(m, d["q"], d["qid"])
        rs.append(ReasoningResult(
            qid=d["qid"], pred=pred, gold=d["gold"],
            correct=numeric_equal(pred, d["gold"]),
        ))
    assert accuracy(rs) == 1.0
    # safety: forbidden import
    assert _safe_exec_block("import os") is None
    return 0


def _demo() -> None:
    """Visible demo: contrast tool-augmented (code+exec) vs pure-CoT on arithmetic.

    Tool path: model emits a Python block, sandbox executes it -> exact answer.
    CoT path:  model only states a guess (no code block) -> run_with_tool yields
               "" and a (deliberately wrong) mental-math guess is scored instead.
    """
    code_model = make_mock_model({
        "ta_1": "```python\nanswer = 17 * 23\n```",
        "ta_2": "```python\nanswer = sum(range(1, 101))\n```",
        "ta_3": "```python\nprimes = [n for n in range(2, 20) "
                "if all(n % d != 0 for d in range(2, n))]\nanswer = len(primes)\n```",
    })
    # A "pure CoT" model that just blurts a (wrong) number, no code block.
    cot_guesses = {"ta_1": "390", "ta_2": "5000", "ta_3": "9"}

    tool_rs, cot_rs = [], []
    print("tool-augmented math (Program-of-Thoughts): model writes code, sandbox runs it")
    for d in _DEMOS:
        tool_pred = run_with_tool(code_model, d["q"], d["qid"])
        # CoT: no executable block -> run_with_tool returns ""; score the raw guess.
        cot_pred = run_with_tool(make_mock_model({d["qid"]: cot_guesses[d["qid"]]}),
                                 d["q"], d["qid"]) or cot_guesses[d["qid"]]
        tool_ok = numeric_equal(tool_pred, d["gold"])
        cot_ok = numeric_equal(cot_pred, d["gold"])
        tool_rs.append(tool_ok)
        cot_rs.append(cot_ok)
        print(f"  {d['q']:28} gold={d['gold']:>5} | "
              f"tool={tool_pred:>6} ({'OK' if tool_ok else 'X'})  "
              f"CoT={cot_pred:>6} ({'OK' if cot_ok else 'X'})")
    print(f"  accuracy: tool-augmented = {sum(tool_rs)/len(tool_rs):.2f}  "
          f"vs pure-CoT = {sum(cot_rs)/len(cot_rs):.2f}")
    print(f"  sandbox guards: import blocked = {_safe_exec_block('import os') is None}, "
          f"dunder blocked = {_safe_exec_block('answer=(1).__class__') is None}")


if __name__ == "__main__":
    f = _self_test()
    print(f"tool_aug_math.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
