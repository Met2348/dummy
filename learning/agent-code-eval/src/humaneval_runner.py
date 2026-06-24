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


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased HumanEval pass@k estimator from Chen et al. 2021.

    Args:
        n: total sampled completions for one task
        c: completions that pass the hidden tests
        k: number of draws being evaluated
    """
    if not (0 <= c <= n):
        raise ValueError("c must be between 0 and n")
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")
    if n - c < k:
        return 1.0

    fail_prob = 1.0
    for i in range(n - c + 1, n + 1):
        fail_prob *= 1.0 - k / i
    return 1.0 - fail_prob


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
    """Pass@1 + pass@k via repeated sampling and the paper estimator.

    Mock models are usually deterministic, but the aggregation mirrors
    HumanEval: sample n completions, count c correct, then estimate pass@k.
    """
    pass1_values = []
    passk_values = []
    n_tasks = 0
    for d in build_prompts():
        correct = 0
        for _ in range(k):
            text = model(d["prompt"], 256)
            code = extract_code(text)
            err = safe_exec(code, d["tests"])
            correct += int(err is None)
        n_tasks += 1
        pass1_values.append(estimate_pass_at_k(k, correct, 1))
        passk_values.append(estimate_pass_at_k(k, correct, k))
    return {
        "pass@1": sum(pass1_values) / max(1, n_tasks),
        f"pass@{k}": sum(passk_values) / max(1, n_tasks),
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
    # pass@k estimator examples
    assert estimate_pass_at_k(10, 0, 5) == 0.0
    assert estimate_pass_at_k(10, 10, 5) == 1.0
    assert abs(estimate_pass_at_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(estimate_pass_at_k(10, 1, 5) - 0.5) < 1e-12
    return 0


def _demo() -> None:
    """Visible demo: run the real exec-based scorer on empty vs oracle models."""
    print(f"HumanEval micro-set: {len(_TASKS)} tasks (real exec() vs hidden tests)")
    empty = run_humaneval(make_mock_model({}, default=""))
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    oracle = run_humaneval(make_mock_model(refs))
    print(f"  empty-model  pass@1 = {pass_rate(empty):.2f}  "
          f"({empty[0].qid}: passed={empty[0].passed}, err={empty[0].error!r})")
    print(f"  oracle       pass@1 = {pass_rate(oracle):.2f}  "
          f"({oracle[0].qid}: passed={oracle[0].passed})")
    pk = run_passk(make_mock_model(refs), k=4)
    print(f"  oracle pass@k (k=4): pass@1={pk['pass@1']:.2f}  pass@4={pk['pass@4']:.2f}")
    print(f"  estimator check: n=10,c=3,k=1 -> {estimate_pass_at_k(10, 3, 1):.2f} "
          "(= c/n, unbiased Chen 2021).")


if __name__ == "__main__":
    f = _self_test()
    print(f"humaneval_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
