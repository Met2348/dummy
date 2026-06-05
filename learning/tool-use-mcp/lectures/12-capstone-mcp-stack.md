# L12 · Capstone — 手写 MCP server + 3 工具 ⭐

## 任务

> 实现 in-memory MCP server，暴露 3 工具，client discover + invoke 全流程跑通。

## 3 工具

| 工具 | 输入 | 输出 |
|------|------|------|
| `calculator` | expression | number |
| `search_kb` | query | top-3 KB entry |
| `get_time` | timezone | iso8601 |

## 流程

```
1. Client.initialize() → capabilities
2. Client.list_tools() → ["calculator","search_kb","get_time"]
3. Client.call_tool("calculator", {"expression":"6*3"}) → 18
4. Client.call_tool("search_kb", {"query":"MCP"}) → top-3
5. Client.call_tool("get_time", {"tz":"UTC"}) → "2026-06-05T..."
6. 错误 case: Client.call_tool("nonexistent",{}) → error -32602
```

## 退出条件

- [ ] initialize + list_tools + 3× call_tool 全 round-trip
- [ ] capabilities 含 tools key
- [ ] 错误 tool 调用返 JSON-RPC -32602
- [ ] All JSON-RPC 2.0 envelope 合规

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/tool-use-mcp/src'); from capstone_mcp_stack import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 预期输出

```markdown
# MCP Capstone

## Handshake
- protocolVersion: 2024-11-05
- serverInfo: capstone-server 0.1
- capabilities: tools

## Tools discovered
- calculator
- search_kb
- get_time

## Calls (4)
| # | tool | args | result |
|---|------|------|--------|
| 1 | calculator | {"expression":"6*3"} | 18 |
| 2 | search_kb | {"query":"MCP"} | top: ... |
| 3 | get_time | {"tz":"UTC"} | 2026-... |
| 4 | bogus | {} | ERROR -32602 Unknown tool |

## Verdict: [PASS] (4 successful round-trips)
```

## 一句话

> 100 行 MCP 全栈 — server 注册 / capabilities 协商 / 3 工具 / client discover + call — JSON-RPC 2.0 合规。
