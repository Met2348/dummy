"""HumanEval (Chen 2021) — 164 Python tasks, pass@1.

Real bench: each task = function signature + docstring + hidden unit tests.
We ship 5 toy tasks to demonstrate the exec + assertion pattern.
"""
from __future__ import annotations

from typing import Dict, List

from common import CodeResult, ModelFn, extract_code, make_mock_model, pass_rate, safe_exec


_TASKS: List[Dict] = [
    {"qid": "he_1",
     "sig": "def add(a, b):",
     "doc": "Return the sum of two integers.",
     "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
     "ref": "def add(a, b):\n    return a + b"},
    {"qid": "he_2",
     "sig": "def is_even(n):",
     "doc": "Return True if n is even, False otherwise.",
     "tests": "assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True",
     "ref": "def is_even(n):\n    return n % 2 == 0"},
    {"qid": "he_3",
     "sig": "def reverse_str(s):",
     "doc": "Return reverse of a string.",
     "tests": "assert reverse_str('hello') == 'olleh'\nassert reverse_str('') == ''",
     "ref": "def reverse_str(s):\n    return s[::-1]"},
    {"qid": "he_4",
     "sig": "def factorial(n):",
     "doc": "Return n! for non-negative integer n.",
     "tests": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(1) == 1",
     "ref": "def factorial(n):\n    out = 1\n    for i in range(2, n+1):\n        out *= i\n    return out"},
    {"qid": "he_5",
     "sig": "def count_vowels(s):",
     "doc": "Return the number of vowels (aeiouAEIOU) in s.",
     "tests": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0",
     "ref": "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')"},
]


def build_prompts() -> List[Dict]:
    out = []
    for t in _TASKS:
        prompt = (
            f"[qid={t['qid']}]\n"
            f"Complete the function:\n```python\n{t['sig']}\n    \"\"\"{t['doc']}\"\"\"\n```\n"
            "Return only the complete function in a code block."
        )
        out.append({"qid": t["qid"], "prompt": prompt,
                    "tests": t["tests"], "ref": t["ref"]})
    return out


def run_humaneval(model: ModelFn) -> List[CodeResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 256)
        code = extract_code(text)
        err = safe_exec(code, d["tests"])
        rs.append(CodeResult(
            qid=d["qid"], pred_code=code, passed=(err is None), error=err,
        ))
    return rs


def run_passk(model: ModelFn, k: int = 4) -> Dict:
    """Pass@1 + pass@k via repeated sampling.

    Mock model is deterministic, so all k draws agree.
    """
    n_correct_at_1 = 0
    n_any_correct = 0
    n_tasks = 0
    for d in build_prompts():
        results_for_task = []
        for _ in range(k):
            text = model(d["prompt"], 256)
            code = extract_code(text)
            err = safe_exec(code, d["tests"])
            results_for_task.append(err is None)
        n_tasks += 1
        if results_for_task[0]:
            n_correct_at_1 += 1
        if any(results_for_task):
            n_any_correct += 1
    return {
        "pass@1": n_correct_at_1 / max(1, n_tasks),
        f"pass@{k}": n_any_correct / max(1, n_tasks),
    }


def _self_test() -> int:
    # Empty model = no code → 0%
    rs = run_humaneval(make_mock_model({}, default=""))
    assert pass_rate(rs) == 0.0
    # Reference solutions wrapped in code block
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    oracle = make_mock_model(refs)
    rs2 = run_humaneval(oracle)
    assert pass_rate(rs2) == 1.0, [r.error for r in rs2]
    # passk
    pk = run_passk(oracle, k=4)
    assert pk["pass@1"] == 1.0 and pk["pass@4"] == 1.0
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"humaneval_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
