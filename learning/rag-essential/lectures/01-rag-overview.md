# L01 · RAG 是什么

## 30 秒定义

> RAG (Retrieval-Augmented Generation) = **retrieve** docs by query → **augment** prompt → **generate** answer。

由 Lewis et al. 2020 提出（NeurIPS），核心思想：

```
Query → Retriever → top-k docs → "Use these docs to answer: {query}" → LLM → Answer
```

## 为什么需要 RAG

| 问题 | 不用 RAG 怎么办 | RAG 解 |
|------|----------------|---------|
| 时效（知识更新） | 重训 | 替换文档 |
| 私域知识 | 微调 | 注入文档 |
| Hallucination | 提示工程 | 让 LLM 引用文档 |
| 可解释 | 黑盒 | 引用源 |
| Token 成本 | 全文塞 | 只塞 top-k |

## RAG 三段式

```
       ┌─────────────────────┐
Query →│ 1. Retrieve (索引)  │
       │  - chunk            │
       │  - embed            │
       │  - top-k search     │
       └─────────┬───────────┘
                 ↓
       ┌─────────────────────┐
       │ 2. Augment (拼装)   │
       │  - rerank           │
       │  - format prompt    │
       └─────────┬───────────┘
                 ↓
       ┌─────────────────────┐
       │ 3. Generate (生成)  │
       │  - LLM call         │
       │  - cite sources     │
       └─────────────────────┘
```

## 2020 vs 2024 vs 2026

| 年 | 代表 | 关键 |
|----|------|------|
| 2020 | Naive RAG (Lewis) | dense retrieval + BART |
| 2023 | LangChain / LlamaIndex 起 | 工程化 |
| 2023.04 | HyDE | hypothesis embed |
| 2024.04 | GraphRAG (MS) | 图结构 |
| 2024.05 | HippoRAG | PageRank |
| 2024-2025 | Contextual retrieval (Anthropic) | chunk + context prefix |
| 2026 | Agentic RAG | RAG-as-tool + multi-step |

## 5 大 advanced RAG 范式

| 范式 | 解决什么 |
|------|---------|
| Hybrid (BM25+vector) | 长尾术语/缩写 |
| Reranker | top-k 精度 |
| HyDE | query/doc 语义 gap |
| GraphRAG | 跨文档关系 |
| Self-RAG / CRAG | 检索质量自评估 |

## 退出条件

- 能讲 RAG 三段式
- 能列 5 RAG 优势（vs 全文塞 / 微调）
- 知道 RAG 2020/2024/2026 三代

## 一句话

> RAG = 给 LLM 一本"开卷小抄"——动态查、动态拼、动态答，便宜可解释。
