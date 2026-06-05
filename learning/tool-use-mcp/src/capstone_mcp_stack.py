"""Capstone — full MCP stack: server with 3 tools + client discover + 4 calls."""
from __future__ import annotations
from mcp_server import MCPServer
from mcp_client import MCPClient


KB = {
    "MCP": "Model Context Protocol — JSON-RPC 2.0 standard from Anthropic (2024.11).",
    "A2A": "Agent-to-Agent protocol — Google 2025 interop standard.",
    "ReAct": "Thought-Action-Observation loop (Yao 2022).",
    "GraphRAG": "Microsoft 2024 — entity graph + community summary RAG.",
}


def _calc(args: dict):
    expr = args.get("expression", "")
    import re
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        raise ValueError(f"unsafe expression: {expr!r}")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307


def _search_kb(args: dict):
    q = args.get("query", "").lower()
    hits = [(k, v) for k, v in KB.items() if q in k.lower() or q in v.lower()]
    if not hits:
        return f"(no KB entry for {q!r})"
    return " | ".join(f"{k}: {v[:60]}" for k, v in hits[:3])


def _get_time(args: dict):
    tz = args.get("tz", "UTC")
    return f"2026-06-05T12:00:00 {tz} (mock fixed time)"


def build_server() -> MCPServer:
    srv = MCPServer("capstone-server")
    srv.add_tool(
        "calculator", "Compute math expression",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        _calc,
    )
    srv.add_tool(
        "search_kb", "Search internal knowledge base",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        _search_kb,
    )
    srv.add_tool(
        "get_time", "Get current time",
        {"type": "object", "properties": {"tz": {"type": "string"}}},
        _get_time,
    )
    return srv


def run_capstone() -> dict:
    srv = build_server()
    client = MCPClient(srv)

    init = client.initialize()
    tools = client.list_tools()

    calls = []
    for name, args in [
        ("calculator", {"expression": "6 * 3"}),
        ("search_kb", {"query": "MCP"}),
        ("get_time", {"tz": "UTC"}),
        ("bogus_tool", {}),
    ]:
        ok, text = client.call_tool(name, args)
        calls.append({"tool": name, "args": args, "ok": ok, "result": text})

    return {
        "init": init,
        "tools": [t["name"] for t in tools],
        "calls": calls,
    }


def to_md(result: dict) -> str:
    lines = [
        "# MCP Capstone — Topic 3 tool-use-mcp\n",
        "## Handshake",
        f"- protocolVersion: {result['init']['protocolVersion']}",
        f"- serverInfo: {result['init']['serverInfo']['name']} {result['init']['serverInfo']['version']}",
        f"- capabilities: tools",
        "\n## Tools discovered",
    ]
    for t in result["tools"]:
        lines.append(f"- {t}")
    lines.append("\n## Calls")
    lines.append("| # | Tool | Args | OK | Result |")
    lines.append("|---|------|------|----|--------|")
    for i, c in enumerate(result["calls"], start=1):
        ok = "[OK]" if c["ok"] else "[FAIL]"
        lines.append(f"| {i} | {c['tool']} | {c['args']} | {ok} | {c['result'][:60]} |")

    success_count = sum(1 for c in result["calls"][:3] if c["ok"])
    error_caught = not result["calls"][3]["ok"]
    verdict = "[PASS]" if success_count == 3 and error_caught else "[FAIL]"
    lines.append(f"\n## Verdict: {verdict} (3 valid + 1 expected error caught)")
    return "\n".join(lines)


def _self_test() -> None:
    res = run_capstone()
    assert res["init"]["serverInfo"]["name"] == "capstone-server"
    assert len(res["tools"]) == 3
    assert "calculator" in res["tools"]

    valid_calls = res["calls"][:3]
    assert all(c["ok"] for c in valid_calls), valid_calls
    assert "18" in valid_calls[0]["result"]
    assert "MCP" in valid_calls[1]["result"]
    assert "2026" in valid_calls[2]["result"]

    bogus = res["calls"][3]
    assert not bogus["ok"] and "-32602" in bogus["result"]
    print("[OK] capstone_mcp_stack._self_test passed (3 success + 1 expected error)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
