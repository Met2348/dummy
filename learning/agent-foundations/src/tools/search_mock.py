"""Mock search — returns canned results based on keyword."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Tool, ActionResult


CANNED = {
    "popular llm": "Claude (most cited 2025 LLM, per Stanford AI Index)",
    "best embedding": "text-embedding-3-large (OpenAI, 2024) and voyage-3 (Voyage AI, 2024)",
    "react agent": "ReAct (Yao 2022) — Thought/Action/Observation loop",
    "mcp protocol": "Model Context Protocol (Anthropic, 2024-11)",
    "graphrag": "GraphRAG (Microsoft 2024) — entity-graph + community summary",
}


def _search(args: dict) -> ActionResult:
    q = args.get("input") or args.get("query") or ""
    if not q:
        return ActionResult(ok=False, error="missing query")
    q_lower = str(q).lower()
    for key, val in CANNED.items():
        if key in q_lower:
            return ActionResult(ok=True, value=val)
    return ActionResult(ok=True, value=f"(no results found for {q!r})")


search_mock_tool = Tool(
    name="search_mock",
    description="Mock web search. Returns canned result if keyword matches.",
    schema={"input": "string (query)"},
    func=_search,
)


def _self_test() -> None:
    r = search_mock_tool.func({"input": "2025 most popular LLM"})
    assert r.ok and "Claude" in r.value, r
    r = search_mock_tool.func({"input": "react agent"})
    assert r.ok and "Yao 2022" in r.value, r
    r = search_mock_tool.func({"input": "completely unknown topic"})
    assert r.ok and "no results" in r.value, r
    print("[OK] search_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
