# L09 · Haystack（deepset）

## 30 秒核心

> Haystack = **RAG-focused** agent framework，企业 search 起家，2024 转 LLM agent。

deepset 公司 2019 项目，HuggingFace 系背景。

## 核心抽象：Pipeline

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import AnthropicChatGenerator

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipe.add_component("prompt_builder", PromptBuilder(template="..."))
pipe.add_component("llm", AnthropicChatGenerator())

pipe.connect("retriever.documents", "prompt_builder.documents")
pipe.connect("prompt_builder.prompt", "llm.messages")

result = pipe.run({"retriever": {"query": "What is X?"}})
```

→ 节点 + connect 模型 (类 Airflow DAG)。

## 强项

| 强 | 解释 |
|----|----|
| Enterprise search 沉淀 | 久 |
| RAG pipeline 强 | document store 多 |
| Eval 强 | 内置 benchmark |
| Multi-modal | text + image + audio |

## 弱项

| 弱 | 解释 |
|----|----|
| LLM agent 弱于 LangGraph | DAG 风格非 state machine |
| 文档少 | 学曲线 |
| 改 API 频 | 1.x → 2.x major refactor |

## 适合

| 适合 | 不适合 |
|------|------|
| Enterprise search | 通用 agent |
| HuggingFace 生态 | OpenAI 全栈 |
| 多模态 RAG | 简单 PoC |

## 退出条件

- 知道 Pipeline + connect 模型
- 知道 deepset 公司
- 与 LlamaIndex 区分 (Haystack 偏 enterprise search)

## 一句话

> Haystack = enterprise search 出身的 RAG-focused agent framework — Pipeline DAG + 多模态。
