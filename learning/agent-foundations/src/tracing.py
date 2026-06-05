"""Tracing & replay for agent debugging."""
from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any, Callable
from common import Trace, Step


def trace_to_json(trace: Trace) -> str:
    data = {
        "question": trace.question,
        "steps": [asdict(s) for s in trace.steps],
        "final": trace.final,
        "tokens_in": trace.tokens_in,
        "tokens_out": trace.tokens_out,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def trace_from_json(s: str) -> Trace:
    data = json.loads(s)
    trace = Trace(question=data.get("question", ""))
    trace.final = data.get("final")
    trace.tokens_in = data.get("tokens_in", 0)
    trace.tokens_out = data.get("tokens_out", 0)
    for sd in data.get("steps", []):
        trace.steps.append(Step(**sd))
    return trace


def make_replay_tool(recorded_obs: dict[str, list[str]]) -> dict[str, Callable[[dict], Any]]:
    """Create mock tools that return recorded observations in order."""
    from common import ActionResult, Tool
    idx = {name: 0 for name in recorded_obs}

    def factory(tool_name: str) -> Tool:
        def fn(args: dict) -> ActionResult:
            i = idx[tool_name]
            idx[tool_name] = min(i + 1, len(recorded_obs[tool_name]) - 1)
            return ActionResult(ok=True, value=recorded_obs[tool_name][i])
        return Tool(name=tool_name, description="replay", schema={}, func=fn)

    return {name: factory(name) for name in recorded_obs}


def cost_summary(trace: Trace, in_price_per_1k: float = 0.003, out_price_per_1k: float = 0.015) -> dict:
    """Approximate USD cost from token counts (Claude Sonnet pricing 2025)."""
    cost_in = trace.tokens_in / 1000.0 * in_price_per_1k
    cost_out = trace.tokens_out / 1000.0 * out_price_per_1k
    return {
        "tokens_in": trace.tokens_in,
        "tokens_out": trace.tokens_out,
        "cost_usd": round(cost_in + cost_out, 6),
        "num_tool_calls": sum(1 for s in trace.steps if s.action != "FINAL" and s.action != "PARSE_FAIL"),
    }


def _self_test() -> None:
    t = Trace(question="Q?")
    t.add(Step(step_num=1, thought="think", action="search('x')", observation="ok"))
    t.final = "done"
    t.tokens_in = 1000
    t.tokens_out = 500

    s = trace_to_json(t)
    t2 = trace_from_json(s)
    assert t2.question == "Q?" and t2.final == "done", t2
    assert len(t2.steps) == 1 and t2.steps[0].observation == "ok"

    summary = cost_summary(t)
    assert summary["tokens_in"] == 1000
    assert summary["num_tool_calls"] == 1
    assert summary["cost_usd"] > 0

    print("[OK] tracing._self_test passed")


if __name__ == "__main__":
    _self_test()
