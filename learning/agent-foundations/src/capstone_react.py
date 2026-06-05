"""Capstone — ReAct loop with 4 mock tools.

Task: 'Find the 2025 most popular LLM name, then compute name_length * 3.'
Expected final answer: 18  (Claude → 6 chars → 6*3=18)
"""
from __future__ import annotations
from common import Trace, make_pattern_llm
from react_loop import react_loop
from tools import ALL_TOOLS
from tracing import cost_summary


CAPSTONE_QUESTION = (
    "Find the 2025 most popular LLM name, then compute its name_length times 3."
)


# Mock LLM scripted to use search_mock + calculator.
CAPSTONE_LLM_RULES = [
    # First reasoning round → search
    (r"Thought 1:",
     "Thought: I need to find the 2025 most popular LLM first.\n"
     "Action 1: search_mock(\"2025 most popular LLM\")\n"),
    # Second reasoning round → compute 6 * 3
    (r"Thought 2:",
     "Thought: Claude has 6 characters, so 6*3.\n"
     "Action 2: calculator(\"6 * 3\")\n"),
    # Third reasoning round → final
    (r"Thought 3:",
     "Thought: Got 18, final.\n"
     "Final Answer: 18\n"),
    # Fallback
    (r".*", "Final Answer: 18\n"),
]


def run_capstone() -> Trace:
    llm = make_pattern_llm(CAPSTONE_LLM_RULES)
    return react_loop(CAPSTONE_QUESTION, llm, ALL_TOOLS, max_steps=6)


def to_md(trace: Trace) -> str:
    lines = [
        "# ReAct Capstone — agent-foundations\n",
        f"**Question:** {trace.question}\n",
        "## Trace\n",
        "| # | Thought | Action | Observation |",
        "|---|---------|--------|-------------|",
    ]
    for s in trace.steps:
        thought = s.thought.replace("\n", " ")[:60]
        action = s.action[:50]
        obs = s.observation.replace("\n", " ")[:60]
        lines.append(f"| {s.step_num} | {thought} | {action} | {obs} |")

    cost = cost_summary(trace)
    lines.append(f"\n## Final answer: **{trace.final}**\n")
    lines.append("## Cost\n")
    lines.append(f"- tokens_in: {cost['tokens_in']}")
    lines.append(f"- tokens_out: {cost['tokens_out']}")
    lines.append(f"- tool_calls: {cost['num_tool_calls']}")
    lines.append(f"- ~cost_usd: {cost['cost_usd']}")

    pass_str = "[PASS]" if trace.final == "18" else "[FAIL]"
    lines.append(f"\n## Verdict: {pass_str}")
    return "\n".join(lines)


def _self_test() -> None:
    trace = run_capstone()
    assert trace.final == "18", trace
    # 至少 2 tool calls (search + calc)
    tool_calls = [s for s in trace.steps if s.action not in ("FINAL", "PARSE_FAIL")]
    assert len(tool_calls) >= 2, tool_calls
    actions = {s.action.split("(")[0] for s in tool_calls}
    assert "search_mock" in actions, actions
    assert "calculator" in actions, actions
    print("[OK] capstone_react._self_test passed (final=18, used 2+ tools)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
