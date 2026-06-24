"""LiveCodeBench (2024) — monthly rolling code benchmark.

Real bench: 400+ problems from LeetCode/Codeforces/AtCoder, monthly
release. Top: GPT-4o ~45%, Claude 3.7 ~52%, o1 ~60%.

Mock: 3 algo problems with hidden tests.
"""
from __future__ import annotations

from typing import Dict, List

from common import CodeResult, ModelFn, extract_code, make_mock_model, pass_rate, safe_exec


_TASKS: List[Dict] = [
    {"qid": "lcb_1", "month": "2025-04",
     "task": "Write `two_sum(nums, target)` returning indices [i, j] of two numbers summing to target.",
     "tests": "assert two_sum([2,7,11,15], 9) == [0,1]\nassert two_sum([3,3], 6) == [0,1]",
     "ref": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target-n], i]\n        seen[n] = i"},
    {"qid": "lcb_2", "month": "2025-05",
     "task": "Write `valid_parens(s)` returning True iff brackets match.",
     "tests": "assert valid_parens('()[]{}') == True\nassert valid_parens('([)]') == False",
     "ref": "def valid_parens(s):\n    stk=[]\n    pairs={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{':\n            stk.append(c)\n        elif c in ')]}':\n            if not stk or stk.pop()!=pairs[c]:\n                return False\n    return not stk"},
    {"qid": "lcb_3", "month": "2025-06",
     "task": "Write `max_subarray(nums)` returning max contiguous subarray sum (Kadane).",
     "tests": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert max_subarray([5,4,-1,7,8]) == 23",
     "ref": "def max_subarray(nums):\n    cur=best=nums[0]\n    for n in nums[1:]:\n        cur = max(n, cur+n)\n        best = max(best, cur)\n    return best"},
]


def build_prompts() -> List[Dict]:
    return [{
        "qid": t["qid"],
        "prompt": (f"[qid={t['qid']}] (Month: {t['month']})\n"
                   f"{t['task']}\nReturn complete function in code block."),
        "tests": t["tests"], "ref": t["ref"],
    } for t in _TASKS]


def run_livecodebench(model: ModelFn) -> List[CodeResult]:
    rs = []
    for d in build_prompts():
        code = extract_code(model(d["prompt"], 512))
        err = safe_exec(code, d["tests"])
        rs.append(CodeResult(qid=d["qid"], pred_code=code,
                              passed=(err is None), error=err))
    return rs


def _self_test() -> int:
    rs = run_livecodebench(make_mock_model({}, default=""))
    assert pass_rate(rs) == 0.0
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    rs2 = run_livecodebench(make_mock_model(refs))
    assert pass_rate(rs2) == 1.0, [r.error for r in rs2]
    return 0


def _demo() -> None:
    """Visible demo: run the real exec-based scorer on empty vs oracle models."""
    print(f"LiveCodeBench micro-set: {len(_TASKS)} algo tasks "
          f"({', '.join(t['qid'] for t in _TASKS)}) — real exec() vs hidden tests")
    empty = run_livecodebench(make_mock_model({}, default=""))
    refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
    oracle = run_livecodebench(make_mock_model(refs))
    print(f"  empty-model  pass@1 = {pass_rate(empty):.2f}")
    print(f"  oracle       pass@1 = {pass_rate(oracle):.2f}  "
          f"(per-task: {[(r.qid, r.passed) for r in oracle]})")
    print("  -> each candidate is really exec()'d against LeetCode-style asserts.")


if __name__ == "__main__":
    f = _self_test()
    print(f"livecodebench_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
