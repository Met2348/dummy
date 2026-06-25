"""把"沙箱真跑 + rubric 判分"接起来。"""
from __future__ import annotations

from .sandbox import SafeExecError, safe_exec
from .task import HELDOUT, VISIBLE


def run_eval(candidate_src: str, rubric) -> dict:
    """在沙箱里真跑候选，取出 classify，用 rubric 判分。返回结构化结果。"""
    try:
        ns, stdout = safe_exec(candidate_src)
    except SafeExecError as e:
        return {"ok": False, "error": str(e), "score": 0.0, "passed": False, "stdout": ""}

    fn = ns.get("classify")
    if not callable(fn):
        return {"ok": False, "error": "未定义 classify", "score": 0.0,
                "passed": False, "stdout": stdout}

    try:
        score, passed = rubric(fn, stdout, VISIBLE, HELDOUT)
    except Exception as e:
        return {"ok": False, "error": f"rubric error: {e}", "score": 0.0,
                "passed": False, "stdout": stdout}
    return {"ok": True, "error": None, "score": float(score),
            "passed": bool(passed), "stdout": stdout}
