# Topic 3: Tool Use & MCP（工具协议化）

> Module 7 第 3 专题 · 12 lectures · ~12h
>
> OpenAI function calling → **Anthropic MCP** (2024.11) → **Google A2A** (2025) → Computer Use → Sandbox 安全执行

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Tool calling 范式 | (intro) |
| L02 | OpenAI function calling | `openai_tools.py` |
| L03 | **MCP** 协议 ⭐ | `mcp_protocol.py` |
| L04 | MCP server impl | `mcp_server.py` |
| L05 | MCP client impl | `mcp_client.py` |
| L06 | **A2A** 协议 ⭐ | `a2a_minimal.py` |
| L07 | Computer Use | `computer_use_mock.py` |
| L08 | Sandbox / e2b / pyodide | `sandbox_mock.py` |
| L09 | Streaming + interrupt | `streaming_tools.py` |
| L10 | Tool error 重试 | `tool_retry.py` |
| L11 | Tool 安全 (injection) | `tool_injection_demo.py` |
| L12 | **Capstone**: 手写 MCP server + 3 工具 | `capstone_mcp_stack.py` |

## Tags

- `tool-use-mcp` — Module 7 第 3 专题

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/tool-use-mcp/src/tests/test_tools.py
```

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/tool-use-mcp/src'); from capstone_mcp_stack import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 关键文献 / 标准

- OpenAI function calling (2023.06)
- Anthropic MCP spec (2024.11) — JSON-RPC 2.0 over stdio / SSE
- Google A2A protocol (2025)
- Anthropic Computer Use (2024.10)
- e2b sandbox / Pyodide

## 一句话

> Tool 三代演化：OpenAI function calling → MCP (互操作) → A2A (多 agent 互操作) — 手写一遍 100% 跑通。
