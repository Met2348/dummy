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
python learning/tool-use-mcp/src/capstone_mcp_stack.py
```

## 预期输出

先跑内置 `_self_test`（真断言），再打印 markdown 报告：

```text
[OK] capstone_mcp_stack._self_test passed (3 success + 1 expected error)
```

```markdown
# MCP Capstone - Topic 3 tool-use-mcp

## Handshake
- protocolVersion: 2024-11-05
- serverInfo: capstone-server 0.1
- capabilities: tools

## Tools discovered
- calculator
- search_kb
- get_time

## Calls
| # | Tool | Args | OK | Result |
|---|------|------|----|--------|
| 1 | calculator | {'expression': '6 * 3'} | [OK] | 18 |
| 2 | search_kb | {'query': 'MCP'} | [OK] | MCP: Model Context Protocol - JSON-RPC-style tool interface  |
| 3 | get_time | {'tz': 'UTC'} | [OK] | 2026-06-05T12:00:00 UTC (mock fixed time) |
| 4 | bogus_tool | {} | [FAIL] | -32602 Unknown tool: bogus_tool |

## Verdict: [PASS] (3 valid + 1 expected error caught)
```

## 一句话

> 100 行 MCP 全栈 — server 注册 / capabilities 协商 / 3 工具 / client discover + call — JSON-RPC 2.0 合规。
