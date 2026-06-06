# L03 · DRA 架构设计

## 我们 DRA 完整流程

```
   User query
       ↓
   ┌─────────┐
   │ Planner │  ← LLM
   └────┬────┘
        ↓ plan = [sub_q_1, sub_q_2, ...]
   ┌─────────┐
   │   for   │
   │ each q  │
   └────┬────┘
        ↓
   ┌─────────────┐
   │  Retriever  │ → tools: search + fetch + cite
   │ (1 q → docs)│
   └─────┬───────┘
        ↓ notes per q
   ┌─────────┐
   │ Writer  │  ← LLM
   └────┬────┘
        ↓ markdown draft
   ┌─────────┐
   │Verifier │  ← LLM (or rule)
   │ claim→src│
   └────┬────┘
        ↓
   Final report (md + [1]-[N])
```

## 模块接口

```python
@dataclass
class Plan:
    sub_questions: list[str]
    rationale: str

@dataclass
class Note:
    sub_question: str
    findings: list[dict]   # {doc_id, snippet, url}

@dataclass
class Draft:
    markdown: str
    claims: list[str]
    citations: list[dict]  # [{ id, source }]

@dataclass
class Verified:
    final_md: str
    supported: list[str]
    unsupported: list[str]
```

## State machine

LangGraph-style，可恢复：

```python
graph = StateGraph(DRAState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("write", write_node)
graph.add_node("verify", verify_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "retrieve")
graph.add_conditional_edges(
    "retrieve",
    lambda s: "retrieve" if s.has_more else "write",
)
graph.add_edge("write", "verify")
graph.add_edge("verify", END)
```

## 失败模式 + 缓解

| 模式 | 缓解 |
|------|------|
| Plan 太宽 | step cap |
| Retrieve 找不到 | fallback web mock |
| Write hallucinate | verify 强制 cite |
| Verifier 误杀 | 多 source 判 supported |
| Cost 爆炸 | budget cap |

## 设计原则

| 原则 | 做 |
|------|---|
| Single responsibility | 每模块一件事 |
| Stateless 通信 | 通过 Plan/Note/Draft 传 |
| Failure local | retriever 失败不爆全流程 |
| Observable | 每步 trace |

## 退出条件

- 能默写 4 模块图
- 能列 4 模块 interface 类型
- 能讲 5 失败模式

## 一句话

> DRA 架构 = Planner → Retriever (loop) → Writer → Verifier — 单一职责 + stateless 通信 + 可观察。
