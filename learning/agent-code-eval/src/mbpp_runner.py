"""MBPP (Austin 2021) — Mostly Basic Python Problems, 974 tasks.

Real bench: each task = NL description + 3 test cases. Our toy version
shows the format.
"""
from __future__ import annotations

from typing import Dict, List

from common import CodeResult, ModelFn, extract_code, make_mock_model, pass_rate, safe_exec


_TASKS: List[Dict] = [
    {"qid": "mbpp_1",
     "task": "Write a function `max_of_three(a, b, c)` that returns the maximum.",
     "tests": "assert max_of_three(1, 2, 3) == 3\nassert max_of_three(5, 1, 2) == 5",
     "ref": "def max_of_three(a, b, c):\n    return max(a, b, c)"},
    {"qid": "mbpp_2",
     "task": "Write a function `is_palindrome(s)` to check palindrome.",
     "tests": "assert is_palindrome('aba') == True\nassert is_palindrome('abc') == False",
     "ref": "def is_palindrome(s):\n    return s == s[::-1]"},
    {"qid": "mbpp_3",
     "task": "Write a function `sum_squares(n)` returning sum of squares 1..n.",
     "tests": "assert sum_squares(3) == 14\nassert sum_squares(0) == 0",
     "ref": "def sum_squares(n):\n    return sum(i*i for i in range(1, n+1))"},
    {"qid": "mbpp_4",
     "task": "Write a function `flatten(lst)` flattening one level of nested list.",
     "tests": "assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]",
     "ref": "def flatten(lst):\n    out = []\n    for sub in lst:\n        out.extend(sub)\n    return out"},
]


def build_prompts() -> List[Dict]:
    return [{
        "qid": t["qid"],
        "prompt": (f"[qid={t['qid']}]\n"
                   f"{t['task']}\nReturn complete function in code block."),
        "tests": t["tests"], "ref": t["ref"],
    } for t in _TASKS]


def run_mbpp(model: ModelFn) -> List[CodeResult]:
    rs = []
    for d in build_prompts():
        code = extract_code(model(d["prompt"], 256))
        err = safe_exec(code, d["tests"])
        rs.append(CodeResult(qid=d["qid"], pred_code=code,
                              passed=(err is None), error=err))
    return rs


def _self_test() -> int:
    rs = run_mbpp(make_mock_model({}, default=""))
    assert pass_rate(rs) == 0.0
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    rs2 = run_mbpp(make_mock_model(refs))
    assert pass_rate(rs2) == 1.0, [r.error for r in rs2]
    return 0


def _demo() -> None:
    """Visible demo: run the real exec-based scorer on empty vs oracle models."""
    print(f"MBPP micro-set: {len(_TASKS)} tasks (real exec() vs test cases)")
    empty = run_mbpp(make_mock_model({}, default=""))
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    oracle = run_mbpp(make_mock_model(refs))
    print(f"  empty-model  pass@1 = {pass_rate(empty):.2f}  "
          f"({empty[0].qid}: passed={empty[0].passed}, err={empty[0].error!r})")
    print(f"  oracle       pass@1 = {pass_rate(oracle):.2f}  "
          f"({oracle[0].qid}: passed={oracle[0].passed})")
    print("  -> pass@1 computed live: exec(candidate) then exec(asserts); err is None => pass.")


if __name__ == "__main__":
    f = _self_test()
    print(f"mbpp_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
