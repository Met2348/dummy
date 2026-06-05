# Topic 6: Agent Framework Stack（框架横评 + 选型）

> Module 7 第 6 专题 · 12 lectures · ~11h
>
> 10 个 framework 横评 + 决策树 + 同任务 3 framework 对照

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 2025-2026 framework 地图 | (intro) |
| L02 | **LangChain** | `langchain_style.py` |
| L03 | **LangGraph** ⭐ deep | `langgraph_style.py` |
| L04 | **LlamaIndex** | `llamaindex_style.py` |
| L05 | **Pydantic AI** | `pydantic_ai_style.py` |
| L06 | **Vercel AI SDK** | `vercel_ai_style.py` |
| L07 | **Claude Agent SDK** ⭐ | `claude_agent_sdk_style.py` |
| L08 | LlamaStack (Meta) | (lecture) |
| L09 | Haystack | (lecture) |
| L10 | Semantic Kernel (MS) | (lecture) |
| L11 | 选型决策树 | `selection_tree.py` |
| L12 | **Capstone**: 同任务 3 framework | `capstone_same_task.py` |

## Tags

- `agent-framework` — Module 7 第 6 专题

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/agent-framework-stack/src/tests/test_frameworks.py
```

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/agent-framework-stack/src'); from capstone_same_task import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 6 framework 决策树

```
Quick PoC?        → CrewAI / Vercel AI SDK
Type-safe required? → Pydantic AI
Complex state machine? → LangGraph
RAG-heavy?        → LlamaIndex
TS-first / Edge?  → Vercel AI SDK
Anthropic stack?  → Claude Agent SDK
C# / .NET?        → Semantic Kernel
Meta stack?       → LlamaStack
```

## 一句话

> 10 framework 横评 + 同 search+summary 任务 3 实现对照 — 看清 framework 是什么。
