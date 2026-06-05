# L01 · 2025-2026 Agent Framework Landscape

## 10 + 主流框架

| Framework | 团队 | 语言 | 主打 |
|-----------|------|------|------|
| **LangChain** | LangChain | Python/TS | 综合 (LCEL + chain) |
| **LangGraph** ⭐ | LangChain | Python/TS | 状态机 + multi-agent |
| **LlamaIndex** | LlamaIndex | Python/TS | RAG + 数据 |
| **CrewAI** | joaomdmoura | Python | role-based crew |
| **AutoGen** | Microsoft | Python | conversable agent |
| **Pydantic AI** | Pydantic | Python | type-safe |
| **Vercel AI SDK** | Vercel | TypeScript | Edge-native streaming |
| **Claude Agent SDK** ⭐ | Anthropic | TS | 官方 Anthropic agent |
| **OpenAI Agents SDK** | OpenAI | Python/TS | 官方 OpenAI agent |
| **LlamaStack** | Meta | Python/Kotlin | Meta 全栈 |
| **Haystack** | deepset | Python | RAG-focused |
| **Semantic Kernel** | Microsoft | C#/Python | 企业 .NET 友好 |
| **DSPy** | Stanford | Python | prompt 编程 |
| **Magentic-One** | Microsoft | Python | multi-agent (Topic 4 讲) |

## 主流 6 选

| Framework | GitHub stars | 2025 trend |
|-----------|-------------:|------------|
| LangChain | 90k | 持续主流 |
| **LangGraph** | 8k | 强势上升 ⬆️ |
| LlamaIndex | 36k | 稳 |
| CrewAI | 30k | 上升 |
| AutoGen | 35k | 0.4 重写 |
| Pydantic AI | 6k | 新秀上升 |

## 选型矩阵（场景 × 框架）

| 场景 | 推荐 1 | 备选 |
|------|-------|------|
| 最快 PoC | CrewAI | Vercel AI SDK |
| 复杂 multi-agent | LangGraph | AutoGen |
| RAG 重 | LlamaIndex | LangChain + RAG |
| Type-safe | Pydantic AI | TypeScript framework |
| Streaming UI | Vercel AI SDK | LangChain.js |
| Anthropic 全栈 | Claude Agent SDK | LangGraph |
| OpenAI 全栈 | OpenAI Agents SDK | LangChain |
| 企业 .NET | Semantic Kernel | — |
| Research | DSPy / AutoGen | — |

## 学习曲线

| 难度 | Framework |
|------|-----------|
| ⭐ | CrewAI / Pydantic AI / Vercel AI SDK |
| ⭐⭐ | LlamaIndex / LangChain (basics) |
| ⭐⭐⭐ | LangGraph / AutoGen / OpenAI Agents |
| ⭐⭐⭐⭐ | Semantic Kernel / DSPy |

## 框架的 5 大组件

不管哪家，都有：
1. **Agent abstraction** (Agent / Runnable / Pipeline)
2. **Tool / Function calling** wrapper
3. **Memory / State** 管理
4. **Streaming / async** 支持
5. **Tracing / debug** 集成

→ 学一家，看其他就明白 80%。

## 2026 预测

| 趋势 | 影响 |
|------|------|
| MCP 普及 | 框架间互操作 |
| A2A 普及 | multi-framework agents 互通 |
| LLM 内置 agent 能力 | 减少 framework 需求 |
| Edge / streaming | Vercel AI SDK 类型上升 |
| Type safety | Pydantic AI / TS 框架上升 |
| Tracing 商业化 | LangSmith / W&B Weave / Phoenix 战 |

## 退出条件

- 能列 10 framework
- 能配场景到框架
- 知道 5 组件 = 共同骨架

## 一句话

> 14 framework 但 5 组件相通 — 学一家，懂其他 80%。
