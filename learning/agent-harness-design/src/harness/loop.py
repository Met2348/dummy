"""The agentic loop — the engine that ties every component together.

    while not done and turns left:
        (maybe compact context)
        resp = model(messages)
        if resp is end_turn:   return text
        for each tool call:
            permission check -> dispatch -> append result to context
        (loop guard: stop if no progress)

This single function is what "the harness" fundamentally *is*.
"""
from __future__ import annotations

from .util import est_tokens, preview
from .errors import LoopGuard


def run_loop(model, registry, context, permissions=None, trace=None, tracker=None,
             max_turns: int = 10, compact_at: int | None = None, loop_guard=None):
    guard = loop_guard or LoopGuard()

    for turn in range(max_turns):
        # 1. context budget management
        if context.over_budget() or (compact_at is not None and context.used() >= compact_at):
            reclaimed = context.compact()
            if reclaimed and trace:
                trace.add("ctx", "compact", f"reclaimed {reclaimed} tokens")

        # 2. model call
        tin = sum(est_tokens(m["content"]) for m in context.messages)
        resp = model.respond(context.messages)
        if tracker:
            tracker.add_model(tin, resp.out_tokens())
        if trace:
            trace.add("model", f"turn-{turn}", resp.stop_reason)

        # 3. end_turn -> done
        if resp.stop_reason == "end_turn":
            context.add("assistant", resp.text)
            if trace:
                trace.add("done", "final", preview(resp.text))
            return resp.text

        # 4. tool_use -> execute each call
        context.add("assistant", resp.text, tool_calls=[tc.name for tc in resp.tool_calls])
        signature = ";".join(sorted(tc.signature() for tc in resp.tool_calls))
        tripped = guard.record(signature)

        for tc in resp.tool_calls:
            tool = registry.get(tc.name)
            if permissions is not None and tool is not None:
                decision = permissions.check(tool, tc.args)
                if trace:
                    trace.add("perm", tc.name, f"{decision.action}: {decision.reason}")
                if decision.action == "deny":
                    context.add("tool", {"ok": False, "error": f"permission denied: {decision.reason}"},
                                name=tc.name, tool_call_id=tc.id)
                    continue
            result = registry.dispatch(tc.name, tc.args)
            if tracker:
                tracker.add_tool()
            if trace:
                trace.add("tool", tc.name, preview(result))
            context.add("tool", result, name=tc.name, tool_call_id=tc.id)

        # 5. loop guard
        if tripped:
            if trace:
                trace.add("loop", "guard", "no progress detected — stopping")
            return None

    if trace:
        trace.add("loop", "max-turns", f"hit cap of {max_turns} turns")
    return None


def _self_test() -> None:
    from .model import MockModel, ModelResponse, ToolCall, tool_results_in
    from .tools import ToolRegistry
    from .context import ContextWindow
    from .tracing import Trace, CostTracker
    from .permissions import PermissionManager

    reg = ToolRegistry()
    reg.add("read_config", "read config", lambda: {"seats": 4, "price": 20}, read_only=True)
    reg.add("multiply", "multiply", lambda a, b: a * b)

    def brain(messages):
        res = {m["name"]: m["content"] for m in messages if m.get("role") == "tool"}
        if "read_config" not in res:
            return ModelResponse("read", [ToolCall("read_config")])
        cfg = res["read_config"]["value"]
        if "multiply" not in res:
            return ModelResponse("calc", [ToolCall("multiply", {"a": cfg["seats"], "b": cfg["price"]})])
        return ModelResponse(f"budget is ${res['multiply']['value']}")

    ctx = ContextWindow(budget=10_000)
    ctx.add("user", "compute the budget")
    trace, tracker = Trace(), CostTracker()
    out = run_loop(MockModel(brain), reg, ctx, permissions=PermissionManager("auto"),
                   trace=trace, tracker=tracker, max_turns=8)
    assert out == "budget is $80", out
    assert tracker.model_calls == 3 and tracker.tool_calls == 2, tracker.summary()
    assert trace.count("tool") == 2 and "done" in trace.kinds()
    assert trace.count("perm") == 2

    # readonly blocks the (write) multiply; loop still terminates via the brain.
    reg2 = ToolRegistry()
    reg2.add("read_config", "read", lambda: {"seats": 4, "price": 20}, read_only=True)
    reg2.add("multiply", "calc", lambda a, b: a * b)  # not read-only -> denied in readonly

    def brain2(messages):
        res = {m["name"]: m["content"] for m in messages if m.get("role") == "tool"}
        if "read_config" not in res:
            return ModelResponse("read", [ToolCall("read_config")])
        if "multiply" not in res:
            return ModelResponse("calc", [ToolCall("multiply", {"a": 4, "b": 20})])
        m = res["multiply"]
        return ModelResponse("done" if m.get("ok") else "blocked: cannot compute")

    ctx2 = ContextWindow(budget=10_000)
    ctx2.add("user", "compute")
    trace2 = Trace()
    out2 = run_loop(MockModel(brain2), reg2, ctx2, permissions=PermissionManager("readonly"),
                    trace=trace2, max_turns=8)
    assert out2 == "blocked: cannot compute", out2
    assert any(s.kind == "perm" and "deny" in s.detail for s in trace2.spans)

    # loop guard: a brain that repeats the same call forever is stopped.
    def stubborn(messages):
        return ModelResponse("again", [ToolCall("read_config")])

    ctx3 = ContextWindow(budget=10_000)
    ctx3.add("user", "go")
    trace3 = Trace()
    out3 = run_loop(MockModel(stubborn), reg, ctx3, max_turns=20, trace=trace3)
    assert out3 is None and "loop" in trace3.kinds()
    print("[OK] harness.loop._self_test passed")


if __name__ == "__main__":
    _self_test()
