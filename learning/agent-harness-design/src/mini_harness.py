"""mini_harness — assemble the components into a usable Harness.

This is the "engine" a product like Claude Code / Cursor wraps around a model:
system prompt + context window + tool dispatch + permissions + tracing, driven
by the agentic loop. Swap MockModel for a real LLM client and the shape is the
same.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/ on path for direct run

from harness.context import ContextWindow
from harness.tracing import Trace, CostTracker
from harness.tools import ToolRegistry
from harness.permissions import PermissionManager
from harness.system_prompt import build_system_prompt
from harness.loop import run_loop


class Harness:
    def __init__(self, model, registry: ToolRegistry, *, role="You are a capable assistant.",
                 env=None, budget: int = 400, permissions=None, max_turns: int = 10,
                 compact_at: int | None = None):
        self.model = model
        self.registry = registry
        self.role = role
        self.env = env or {}
        self.budget = budget
        self.permissions = permissions or PermissionManager(mode="auto")
        self.max_turns = max_turns
        self.compact_at = compact_at
        self.trace = Trace()
        self.tracker = CostTracker()
        self.context: ContextWindow | None = None

    def run(self, goal: str):
        ctx = ContextWindow(budget=self.budget)
        system = build_system_prompt(self.role, list(self.registry.tools.values()), self.env)
        ctx.add("system", system)
        self.trace.add("system", "prompt", f"{len(self.registry.names())} tools, "
                                            f"{len(self.env)} env vars")
        ctx.add("user", goal)
        result = run_loop(self.model, self.registry, ctx, permissions=self.permissions,
                          trace=self.trace, tracker=self.tracker,
                          max_turns=self.max_turns, compact_at=self.compact_at)
        self.context = ctx
        return result

    def report(self) -> str:
        return (self.trace.render() + "\n\ncost: " + str(self.tracker.summary())
                + f"\npermissions: {self.permissions.log}")


def _self_test() -> None:
    from harness.model import MockModel, ModelResponse, ToolCall

    reg = ToolRegistry()
    reg.add("now", "return a constant timestamp", lambda: "T0", read_only=True)

    def brain(messages):
        res = {m["name"]: m["content"] for m in messages if m.get("role") == "tool"}
        if "now" not in res:
            return ModelResponse("checking time", [ToolCall("now")])
        return ModelResponse(f"the time is {res['now']['value']}")

    h = Harness(MockModel(brain), reg, role="You are a clock.", env={"tz": "UTC"})
    out = h.run("what time is it?")
    assert out == "the time is T0", out
    assert h.trace.kinds()[0] == "system"
    assert "model" in h.trace.kinds() and "tool" in h.trace.kinds() and "done" in h.trace.kinds()
    assert h.tracker.model_calls == 2 and h.tracker.tool_calls == 1
    assert "cost:" in h.report()
    print("[OK] mini_harness._self_test passed")


if __name__ == "__main__":
    _self_test()
