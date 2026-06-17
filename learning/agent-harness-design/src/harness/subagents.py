"""Sub-agents — spawning an isolated nested loop and aggregating its result.

A sub-agent runs in its OWN context window (isolation), so a big, noisy subtask
(deep search, multi-file read) doesn't pollute the parent's window. The parent
gets back only a compact summary. This is the harness-level building block
behind "fan out to workers" / orchestrator-workers (Topic 8 L07).
"""
from __future__ import annotations

from .context import ContextWindow
from .tracing import Trace, CostTracker
from .loop import run_loop


def run_subagent(goal: str, model, registry, *, budget: int = 2000, max_turns: int = 6,
                 permissions=None) -> dict:
    """Run a nested agent loop with an isolated context; return a compact summary."""
    ctx = ContextWindow(budget=budget)
    ctx.add("user", goal)
    trace, tracker = Trace(), CostTracker()
    result = run_loop(model, registry, ctx, permissions=permissions,
                      trace=trace, tracker=tracker, max_turns=max_turns)
    return {
        "goal": goal,
        "result": result,
        "ok": result is not None,
        "model_calls": tracker.model_calls,
        "tool_calls": tracker.tool_calls,
        "context_msgs": len(ctx.messages),  # stays in the sub-agent, not the parent
    }


def fan_out(goals, model_factory, registry, **kw) -> list:
    """Run several sub-agents (logically parallel) and collect their summaries.
    `model_factory(goal) -> model` lets each sub-agent get its own brain."""
    return [run_subagent(g, model_factory(g), registry, **kw) for g in goals]


def _self_test() -> None:
    from .model import MockModel, ModelResponse, ToolCall
    from .tools import ToolRegistry

    reg = ToolRegistry()
    reg.add("lookup", "look up a fact", lambda topic: f"fact about {topic}", read_only=True)

    def make_brain(goal):
        def brain(messages):
            res = {m["name"]: m["content"] for m in messages if m.get("role") == "tool"}
            if "lookup" not in res:
                return ModelResponse("looking up", [ToolCall("lookup", {"topic": goal})])
            return ModelResponse(f"summary: {res['lookup']['value']}")
        return brain

    one = run_subagent("pricing", MockModel(make_brain("pricing")), reg)
    assert one["ok"] and "pricing" in one["result"]
    assert one["model_calls"] == 2 and one["tool_calls"] == 1

    many = fan_out(["a", "b", "c"], lambda g: MockModel(make_brain(g)), reg)
    assert len(many) == 3 and all(s["ok"] for s in many)
    # Each sub-agent keeps its own context (isolation) — parent never sees it.
    assert all(s["context_msgs"] >= 3 for s in many)
    print("[OK] harness.subagents._self_test passed")


if __name__ == "__main__":
    _self_test()
