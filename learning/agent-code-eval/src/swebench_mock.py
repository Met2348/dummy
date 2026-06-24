"""SWE-Bench Verified (Princeton 2024) — mock with 1 fake issue.

Real SWE-Bench: 2294 real GitHub issues + test patches. Agent must
modify multiple files to make hidden test pass. Top: ~50% (Claude 3.7).

Our mock: 1 fake repo with 1 buggy function, 1 unit test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from common import ModelFn, extract_code, make_mock_model, safe_exec


# === Mock repo state ===

REPO_FILE_BUG = """def divide(a, b):
    return a / b
"""

REPO_FILE_FIXED = """def divide(a, b):
    if b == 0:
        raise ValueError('cannot divide by zero')
    return a / b
"""

REPO_TESTS = """try:
    divide(10, 0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert divide(10, 2) == 5
"""


@dataclass
class SweTask:
    qid: str
    issue: str
    buggy_file: str
    hidden_tests: str


_TASKS: List[SweTask] = [
    SweTask(
        qid="swe_1",
        issue=("Issue #42: `divide(a, 0)` returns inf/raises ZeroDivisionError. "
               "Should raise ValueError with message 'cannot divide by zero'."),
        buggy_file=REPO_FILE_BUG,
        hidden_tests=REPO_TESTS,
    ),
]


def build_prompts() -> List[Dict]:
    out = []
    for t in _TASKS:
        prompt = (
            f"[qid={t.qid}]\n"
            f"Repository file (utils.py):\n```python\n{t.buggy_file}\n```\n"
            f"Issue: {t.issue}\n"
            "Provide the fixed file contents in a code block."
        )
        out.append({"qid": t.qid, "prompt": prompt, "tests": t.hidden_tests})
    return out


def run_swebench_mock(model: ModelFn) -> List[Dict]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 512)
        code = extract_code(text)
        err = safe_exec(code, d["tests"])
        rs.append({
            "qid": d["qid"], "pred_code": code,
            "passed": err is None, "error": err,
        })
    return rs


def _self_test() -> int:
    # Empty model fails
    rs = run_swebench_mock(make_mock_model({}, default="x = 1"))
    assert not rs[0]["passed"]
    # Fix passes
    rs2 = run_swebench_mock(make_mock_model({
        "swe_1": f"```python\n{REPO_FILE_FIXED}```",
    }))
    assert rs2[0]["passed"], rs2[0]["error"]
    return 0


def _demo() -> None:
    """Visible demo: show the fix is graded by really running the hidden test."""
    print(f"SWE-Bench mock: {len(_TASKS)} issue (divide-by-zero) — real exec() of patched file")
    buggy = run_swebench_mock(make_mock_model({"swe_1": f"```python\n{REPO_FILE_BUG}```"}))
    fixed = run_swebench_mock(make_mock_model({"swe_1": f"```python\n{REPO_FILE_FIXED}```"}))
    print(f"  submit BUGGY file -> passed={buggy[0]['passed']}  (err={buggy[0]['error']!r})")
    print(f"  submit FIXED file -> passed={fixed[0]['passed']}")
    print("  -> resolved iff the patched file makes the hidden ValueError test pass "
          "(mechanism, not a constant).")


if __name__ == "__main__":
    f = _self_test()
    print(f"swebench_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
