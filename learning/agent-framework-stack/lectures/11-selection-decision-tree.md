# L11 · 选型决策树

## 决策树（按 use case）

```
你的任务?
├─ 简单 chatbot (1-2 LLM call)
│   └─ Vercel AI SDK (TS) / Pydantic AI (Python)
│
├─ RAG 重度
│   ├─ 入门 → LlamaIndex
│   ├─ 多 strategy → LangChain (LangGraph 编排)
│   └─ Enterprise search → Haystack
│
├─ Multi-agent
│   ├─ Role-based 简单 → CrewAI
│   ├─ 复杂状态 → LangGraph
│   └─ 通用 → AutoGen / Magentic-One
│
├─ Agent 产品 (deep research / code)
│   ├─ Anthropic 全栈 → Claude Agent SDK
│   ├─ OpenAI 全栈 → OpenAI Agents SDK
│   ├─ TS / Next.js → Vercel AI SDK
│   └─ 通用 → LangGraph
│
├─ Type-safe 强需求
│   └─ Pydantic AI (Python) / Zod + Vercel (TS)
│
└─ 企业 Microsoft 栈
    └─ Semantic Kernel
```

## 按团队语言

| 主语言 | 推荐 |
|--------|------|
| Python | LangChain / LangGraph / LlamaIndex |
| TypeScript | Vercel AI SDK / LangChain.js |
| C# / .NET | Semantic Kernel |
| Java / Kotlin | LangChain4j / Semantic Kernel Java |
| Rust | rig-rs (新秀) |
| Go | Eino (字节, 2024) |

## 按规模

| 规模 | 推荐 |
|------|------|
| Solo PoC | CrewAI / Pydantic AI / Vercel AI SDK |
| Small team | LangGraph / LlamaIndex |
| Enterprise | Semantic Kernel / LlamaStack / Haystack |
| Research lab | DSPy / AutoGen |

## 按性能 / latency

| 性能要求 | 推荐 |
|---------|------|
| Edge / streaming | Vercel AI SDK |
| 低 latency | 直接调 LLM API (无 framework) |
| 高 throughput | 自建 stack (frameworks add overhead) |
| Cost-sensitive | DSPy (auto-tune prompts) |

## 按 LLM 厂

| LLM | 官方 framework |
|-----|---------------|
| Anthropic Claude | Claude Agent SDK |
| OpenAI GPT | OpenAI Agents SDK |
| Meta Llama | LlamaStack |
| Microsoft Azure | Semantic Kernel |

→ 跨 LLM 用 LangChain / LangGraph。

## 反模式

| 模式 | 反 |
|------|---|
| "all-in-one" framework | 1 个 LLM call 不需框架 |
| 多 framework 混 | 维护噩梦 |
| Framework 重写 (langchain 0.x → 0.3) | 锁版 |
| Custom-build everything | 总有 90% 别人做好 |

## 选型 checklist

- [ ] LLM vendor 锁了吗？
- [ ] Multi-agent 需要？
- [ ] State 持久化需要？
- [ ] Streaming UI 需要？
- [ ] Type safety 需要？
- [ ] 企业 compliance 需要？
- [ ] 团队语言？
- [ ] 现有依赖？

## 实现 (`selection_tree.py` 预告)

```python
def select_framework(
    multi_agent: bool,
    rag_heavy: bool,
    typed: bool,
    vendor: str,
    language: str,
    enterprise: bool,
) -> str:
    if enterprise and language == "csharp":
        return "Semantic Kernel"
    if vendor == "anthropic":
        return "Claude Agent SDK"
    if multi_agent:
        return "LangGraph"
    if rag_heavy:
        return "LlamaIndex"
    if typed:
        return "Pydantic AI"
    if language == "typescript":
        return "Vercel AI SDK"
    return "LangChain"
```

## 退出条件

- 能用决策树选 framework
- 能列 6 反模式
- 能写选型 checklist

## 一句话

> Framework 选型 = vendor + language + use-case + 团队规模 4 axis — 决策树 + 反模式 + checklist 三件套不踩坑。
