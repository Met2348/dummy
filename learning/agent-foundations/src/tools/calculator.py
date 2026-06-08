"""Calculator tool - safe arithmetic eval (no builtins)."""
from __future__ import annotations
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Tool, ActionResult


def _safe_eval(expr: str) -> float:
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%]+", expr):
        raise ValueError(f"unsupported chars in: {expr!r}")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307


def _calc(args: dict) -> ActionResult:
    expr = args.get("input") or args.get("expression") or ""
    if not expr:
        return ActionResult(ok=False, error="missing expression")
    try:
        return ActionResult(ok=True, value=_safe_eval(str(expr)))
    except Exception as e:  # noqa: BLE001
        return ActionResult(ok=False, error=str(e))


calculator_tool = Tool(
    name="calculator",
    description="Compute math expression like '2+3*4'.",
    schema={"input": "string (math expression)"},
    func=_calc,
)


def _self_test() -> None:
    r = calculator_tool.func({"input": "2+3"})
    assert r.ok and r.value == 5, r
    r = calculator_tool.func({"input": "(10-2)*3"})
    assert r.ok and r.value == 24, r
    r = calculator_tool.func({"input": "import os"})
    assert not r.ok, r
    print("[OK] calculator._self_test passed")


if __name__ == "__main__":
    _self_test()
