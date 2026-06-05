"""Plan-and-Execute pattern.

Planner LLM lists steps once → Executor runs them in order.
"""
from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from common import Tool, ActionResult


@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    final: str | None = None


def parse_plan(plan_text: str) -> list[str]:
    """Extract numbered steps from planner LLM output."""
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line[:3]:
            steps.append(line.split(".", 1)[1].strip())
    return steps


def plan_execute(
    question: str,
    planner_llm: Callable[[str], str],
    executor_fn: Callable[[str, dict], str],
    final_llm: Callable[[str], str],
) -> Plan:
    plan_text = planner_llm(f"Question: {question}\nPlan as numbered list:")
    steps = parse_plan(plan_text)
    plan = Plan(steps=steps)
    for i, step_desc in enumerate(steps):
        result = executor_fn(step_desc, plan.results)
        plan.results[f"E{i+1}"] = result
    summary = "\n".join(f"E{i+1}: {plan.results[f'E{i+1}']}" for i in range(len(steps)))
    plan.final = final_llm(f"Question:{question}\nObservations:\n{summary}\nFinal:")
    return plan


def make_simple_executor(tools: dict[str, Tool]) -> Callable[[str, dict], str]:
    """Naive executor — keyword match step to tool."""
    def execute(step: str, prior: dict) -> str:
        s = step.lower()
        if "search" in s or "find" in s:
            tool_in = step.replace("Search for", "").replace("search", "").strip(": ")
            r = tools["search_mock"].func({"input": tool_in})
            return r.to_obs()
        if "compute" in s or "calc" in s or "+" in s or "*" in s:
            import re
            m = re.search(r"[\d\+\-\*\/\(\)\s\.]+", step)
            if m:
                r = tools["calculator"].func({"input": m.group(0).strip()})
                return r.to_obs()
        if "read" in s and "file" in s:
            r = tools["file_op"].func({"action": "list", "path": "_"})
            return r.to_obs()
        return f"(no-op for {step!r})"
    return execute


def _self_test() -> None:
    from tools import ALL_TOOLS
    from common import make_pattern_llm

    planner = make_pattern_llm([(r".*", "Plan:\n1. Search for popular LLM\n2. Compute 3+4")])
    final_llm = make_pattern_llm([(r".*", "DONE")])
    plan = plan_execute("X?", planner, make_simple_executor(ALL_TOOLS), final_llm)
    assert len(plan.steps) == 2, plan.steps
    assert "E1" in plan.results and "E2" in plan.results
    assert "7" in plan.results["E2"], plan.results
    assert plan.final == "DONE"
    print("[OK] plan_execute._self_test passed")


if __name__ == "__main__":
    _self_test()
