# L04 · LlamaIndex

## 30 秒核心

> LlamaIndex = **数据 framework** for LLM apps. RAG / index / agent 偏数据侧。

36k stars，2022 项目，前称 GPT Index。

## 核心抽象

| 抽象 | 用途 |
|------|------|
| `Document` | 原文档 |
| `Node` | chunk (有 metadata) |
| `Index` | 数据结构 (Vector/Tree/List/KG) |
| `Retriever` | 找 node |
| `Query Engine` | retrieve + LLM |
| `Agent` | tool-use agent |

## 简单 RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is X?")
print(response.response)
print(response.source_nodes)  # 自动 citation
```

3 行做 RAG，自动 citation。

## 5 种 Index

| Index | 数据结构 | 用 |
|-------|---------|---|
| `VectorStoreIndex` | embedding + DB | 默认 |
| `SummaryIndex` | sequential | 总结全文 |
| `TreeIndex` | 层次 | 长文档导航 |
| `KeywordTableIndex` | 关键词 → node | 精确 |
| `KnowledgeGraphIndex` | triple → node | 关系 |

## Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def search(query): return ...
tool = FunctionTool.from_defaults(fn=search)
agent = ReActAgent.from_tools([tool], llm=llm)
response = agent.chat("Find Tokyo weather")
```

ReAct loop + tool use 一键。

## Workflows (2024)

```python
from llama_index.core.workflow import Workflow, step, Event

class MyFlow(Workflow):
    @step
    async def step_1(self, event: StartEvent) -> Event2:
        return Event2(data="processed")

    @step
    async def step_2(self, event: Event2) -> StopEvent:
        return StopEvent(result="done")
```

Event-driven，与 LangGraph 状态机异。

## 强弱

| 强 | 弱 |
|----|----|
| RAG 体验最佳 | agent 不如 LangGraph 丰富 |
| 5 index 灵活 | abstraction 重 |
| Citation 自动 | 学曲线陡 |
| 数据 connector 丰富 | 单机 RAG 优先 |

## LlamaIndex vs LangChain (RAG 维度)

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| 入门 RAG | 3 行 | 10 行 |
| Citation | 自动 | 手动 |
| Index 类型 | 5 | 主要 vector |
| Agent | ReAct + simple | LangGraph 强 |
| 数据 reader | 200+ | 100+ |

## 我们 mock 版（`llamaindex_style.py` 预告）

```python
class Document: ...
class VectorStoreIndex:
    @classmethod
    def from_documents(cls, docs): ...
    def as_query_engine(self): ...
```

## 退出条件

- 能默写 3 行 RAG
- 能列 5 index
- 与 LangChain 区分 RAG 体验

## 一句话

> LlamaIndex = 数据 framework + RAG 体验最佳 — 3 行 RAG，5 种 index，36k stars。
