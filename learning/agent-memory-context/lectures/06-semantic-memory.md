# L06 · Semantic Memory + Knowledge Graph

## 30 秒

> Semantic memory = **抽象事实 / 概念 / 关系**：
> - "Alice 是 ML 工程师"
> - "Alice 偏好 Anthropic Claude"
> - "Alice 项目 = RAG agent for legal"

## 存储 2 种

### A. Vector + facts（Mem0 风格）

```python
{"fact": "Alice prefers Anthropic Claude", "embedding": [...]}
```

简单，retrieval 用 cosine。

### B. Knowledge Graph

```
(Alice) --is_a--> (ML Engineer)
(Alice) --prefers--> (Anthropic Claude)
(Alice) --works_on--> (RAG-legal-project)
(RAG-legal-project) --uses--> (LangChain)
```

→ 支持 multi-hop query: "Alice 项目用什么 framework?"

## KG 抽取流程

```
Conversation:
  "Alice 在做一个 RAG 项目，用 LangChain。她偏好 Anthropic。"

LLM extract triples:
  (Alice, works_on, RAG_project)
  (RAG_project, uses_framework, LangChain)
  (Alice, prefers, Anthropic)

Insert into KG:
  graph[Alice]["works_on"] = "RAG_project"
  ...
```

## 查询

```python
# 单跳
graph[Alice]["prefers"]  # → Anthropic

# 多跳: Alice 项目用什么 framework?
work = graph[Alice]["works_on"]  # RAG_project
framework = graph[work]["uses_framework"]  # LangChain
```

## Conflict resolution

```python
# 新事实: "Alice 偏好 Gemini"
old = graph[Alice]["prefers"]  # "Anthropic"
if old != "Gemini":
    # 选 1: 覆盖
    graph[Alice]["prefers"] = "Gemini"
    # 选 2: 加版本
    graph[Alice]["prefers"] = [
        {"value":"Anthropic","time":"2025-12"},
        {"value":"Gemini","time":"2026-06"},
    ]
```

## 哪些事实进 semantic

| 进 | 不进 |
|----|------|
| User profile | 一次性闲聊 |
| 长期偏好 | 单 turn fact |
| 工具熟悉度 | 临时计算结果 |
| 项目背景 | 当下问题描述 |

→ Mem0 的 extract LLM 自决"是否值得记"。

## 实现 (`semantic_memory.py` 预告)

```python
class SemanticMemory:
    def __init__(self):
        self.facts: dict[str, list[dict]] = {}  # user_id → facts
        self.graph: dict = {}  # adjacency

    def add_triple(self, subj, pred, obj, user_id):
        self.graph.setdefault(subj, {})[pred] = obj
        self.facts.setdefault(user_id, []).append({
            "triple": (subj, pred, obj),
            "embedding": hash_embed(f"{subj} {pred} {obj}"),
        })

    def query(self, subj, pred=None):
        if pred:
            return self.graph.get(subj, {}).get(pred)
        return self.graph.get(subj, {})

    def search(self, query, user_id, k=5):
        ...
```

## 与 RAG 区别

| RAG | Semantic memory |
|-----|-----------------|
| 文档 chunk | 抽 atomic fact |
| Embedding 检索 | 检索 + 图查询 |
| 不更新 | 主动 update |
| 全局 | per-user |

## 退出条件

- 能讲 vector vs KG 两 store
- 能写 extract triples 流程
- 能讲 conflict 2 种处理

## 一句话

> Semantic memory = 抽象事实 (vector 或 KG) + LLM extract + conflict 解决 — 用户长期 profile 的载体。
