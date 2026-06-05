"""Common helpers for framework mocks."""
from __future__ import annotations
from dataclasses import dataclass, field


MOCK_KB = {
    "react": "ReAct (Yao 2022) — Thought-Action-Observation loop pattern.",
    "rag": "RAG (Lewis 2020) — retrieve-augment-generate pipeline.",
    "mcp": "MCP (Anthropic 2024.11) — Model Context Protocol over JSON-RPC.",
    "graphrag": "GraphRAG (Microsoft 2024) — entity graph + community summary.",
    "langgraph": "LangGraph — StateGraph + reducers + checkpoint.",
}


def mock_search(query: str, k: int = 3) -> list[str]:
    q = query.lower()
    hits = [v for key, v in MOCK_KB.items() if key in q or q in key]
    if not hits:
        hits = list(MOCK_KB.values())
    return hits[:k]


def mock_summarize(query: str, contexts: list[str]) -> str:
    if not contexts:
        return f"No info found about {query}."
    head = contexts[0][:120]
    return f"Summary about {query}: {head}"


@dataclass
class FrameworkRun:
    framework: str
    loc: int
    output: str
    abstraction_level: str = ""


def _self_test() -> None:
    results = mock_search("ReAct agent")
    assert len(results) >= 1
    assert any("ReAct" in r for r in results), results

    summary = mock_summarize("ReAct", results)
    assert "Summary about" in summary
    assert "ReAct" in summary

    empty = mock_summarize("X", [])
    assert "No info" in empty
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
