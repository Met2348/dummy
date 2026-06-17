"""Tool execution layer — registry, dispatch, structured results.

Every tool returns a structured {"ok", "value"|"error"} envelope so errors are
*surfaced* into the conversation (the model can see and react), never swallowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Tool:
    name: str
    description: str          # the model picks tools by this — keep it crisp
    fn: Callable[..., Any]
    read_only: bool = False   # used by the permission layer

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def add(self, name: str, description: str, fn, read_only: bool = False) -> "ToolRegistry":
        self.tools[name] = Tool(name, description, fn, read_only)
        return self

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def names(self) -> list:
        return list(self.tools)

    def dispatch(self, name: str, args: dict | None = None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return {"ok": True, "value": tool(**(args or {}))}
        except Exception as e:  # noqa: BLE001 — surface, don't swallow
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def dispatch_many(self, calls) -> list:
        """Run several tool calls (logically parallel). Returns [(call, result)]."""
        return [(c, self.dispatch(c.name, c.args)) for c in calls]


def _self_test() -> None:
    reg = ToolRegistry()
    reg.add("calc", "add two ints", lambda a, b: a + b)
    reg.add("read", "read a key", lambda key: {"k": key}, read_only=True)

    ok = reg.dispatch("calc", {"a": 2, "b": 3})
    assert ok == {"ok": True, "value": 5}, ok
    assert reg.get("read").read_only is True

    unknown = reg.dispatch("nope", {})
    assert (not unknown["ok"]) and "unknown tool" in unknown["error"]

    bad = reg.dispatch("calc", {"a": 1})  # missing arg -> TypeError surfaced
    assert (not bad["ok"]) and "TypeError" in bad["error"]

    from .model import ToolCall
    many = reg.dispatch_many([ToolCall("calc", {"a": 1, "b": 1}), ToolCall("read", {"key": "x"})])
    assert len(many) == 2 and many[0][1]["value"] == 2
    assert reg.names() == ["calc", "read"]
    print("[OK] harness.tools._self_test passed")


if __name__ == "__main__":
    _self_test()
