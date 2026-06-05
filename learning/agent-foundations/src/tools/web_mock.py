"""Mock web fetch — returns canned page content by URL."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Tool, ActionResult


CANNED_PAGES = {
    "https://anthropic.com/mcp": "MCP unifies tool/resource/prompt as JSON-RPC primitives.",
    "https://langchain.com/langgraph": "LangGraph: stateful agent applications with StateGraph.",
    "https://arxiv.org/abs/2210.03629": "ReAct paper abstract — Yao 2022.",
}


def _web(args: dict) -> ActionResult:
    url = args.get("input") or args.get("url") or ""
    if not url:
        return ActionResult(ok=False, error="missing url")
    if url in CANNED_PAGES:
        return ActionResult(ok=True, value=CANNED_PAGES[url])
    return ActionResult(ok=True, value=f"(mock 404 for {url})")


web_mock_tool = Tool(
    name="web_mock",
    description="Mock HTTP GET. Returns canned content for known URLs.",
    schema={"input": "string (url)"},
    func=_web,
)


def _self_test() -> None:
    r = web_mock_tool.func({"input": "https://anthropic.com/mcp"})
    assert r.ok and "MCP" in r.value, r
    r = web_mock_tool.func({"input": "https://unknown.com"})
    assert r.ok and "mock 404" in r.value, r
    print("[OK] web_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
