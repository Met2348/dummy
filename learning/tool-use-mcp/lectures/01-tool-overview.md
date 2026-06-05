# L01 · Tool Calling 范式 — 三代演化

## 时间线

| 年 | 标准 | 团队 |
|---|------|------|
| 2023.06 | **function calling** | OpenAI |
| 2023.10 | tool use API | Anthropic |
| 2024.05 | parallel tool calls | OpenAI |
| **2024.11** | **MCP** (Model Context Protocol) | Anthropic |
| 2024.10 | **Computer Use** | Anthropic |
| **2025** | **A2A** (Agent-to-Agent) | Google |

## 第一代：function calling（2023）

```json
{
  "tools": [{"name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}],
  "tool_choice": "auto"
}
```

LLM 输出：
```json
{"tool_calls":[{"name":"search","arguments":{"q":"X"}}]}
```

**问题**：每家协议不同。OpenAI / Anthropic / Google 互不兼容。

## 第二代：MCP（2024.11）

```
统一标准的 client-server 协议:
- stdio / SSE / WebSocket transport
- JSON-RPC 2.0 envelope
- 三大 primitive: tools / resources / prompts
```

**生态**：Anthropic 主推，2025 上半年 1000+ MCP server (filesystem / git / database / slack / ...)。

## 第三代：A2A（2025）

```
Agent-to-Agent 互操作:
- Agent Card (能力描述)
- skill exchange (调用别的 agent 的 skill)
- task routing
```

Google 主导，2025.04 Beta，对应 multi-agent 互操作。

## 一图三代

```
function calling   →   MCP          →   A2A
"LLM 调 tool"           "client-server"  "agent-agent"
单家协议              统一接入           多 agent 互操作
```

## 范式背后的真问题

| 问题 | 第一代 | MCP | A2A |
|------|-------|-----|-----|
| 单家协议绑定 | ❌ | ✓ | ✓ |
| 工具发现 | ✗ | ✓ tools/list | ✓ skill discovery |
| 跨进程 | ✗ | ✓ stdio | ✓ HTTP |
| 多 agent 互操作 | ✗ | ✗ | ✓ |
| 状态管理 | ✗ | partial | ✓ |

## 退出条件

- 时间线背 3 代
- 能列 3 大 MCP primitive
- 知道 A2A 解决 multi-agent 互操作

## 一句话

> Tool 三代演化：function calling 是私有 API，MCP 是公共接入，A2A 是 agent 间联邦。
