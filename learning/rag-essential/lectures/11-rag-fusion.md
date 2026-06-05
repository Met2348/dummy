# L11 · RAG-Fusion / Multi-Query

## 30 秒核心

> 一个 query 容易 retrieve 不全 → 生成 N 个等价 query → 各 retrieve → RRF 融合。

## Multi-Query Retrieval（LangChain pattern）

```
Q: "What's the impact of LLMs on education?"
   ↓ LLM rewriter
Q1: "How LLMs change classroom?"
Q2: "Effects of AI on teaching methods"
Q3: "ChatGPT in education research"
   ↓ retrieve each
top-k 1, top-k 2, top-k 3
   ↓ RRF
final top-k
```

## RAG-Fusion (Adrian Raudaschl 2023)

Multi-Query + RRF + reranker：

```python
def rag_fusion(q, llm, retriever, k=5):
    queries = llm(f"Generate 4 query rewrites:\n{q}")
    results = [retriever(qq) for qq in queries]
    ranked = rrf(results, k=k*2)
    return reranker(q, ranked, k=k)
```

## 与 HyDE 对比

| 范式 | 改 query | 改 doc |
|------|---------|--------|
| HyDE | ✓ (rewrite as hypo answer) | ✗ |
| Multi-Query | ✓ (multiple rewrites) | ✗ |
| GraphRAG | ✗ | ✓ (build graph) |

## 何时用

| 场景 | 推荐 |
|------|------|
| Query 短抽象 | Multi-Query |
| Query 单一术语 | HyDE |
| Query 复杂 multi-aspect | 二者结合 |

## 数字（LangChain blog 2023）

| Method | recall@10 |
|--------|----------:|
| Single | 65% |
| Multi-Query (3 rewrites) | 75% |
| RAG-Fusion (rewrites + RRF) | 78% |

## 成本

- N rewrites = N+1 LLM call
- N retrieve = N× search
- 一般 N=3-5 足够

## Step-Back Prompting (Zheng 2023)

```
Q: "What was Einstein's job in 1933?"
   ↓ step-back
Step-back Q: "What were Einstein's life events?"
   ↓ retrieve broader
更全 context → final answer
```

## 退出条件

- 能讲 Multi-Query / RAG-Fusion / Step-Back 区别
- 知道 RAG-Fusion = rewrite + RRF + rerank

## 一句话

> Multi-Query / RAG-Fusion = 一问变多问 + RRF 融合 —— 简单粗暴提 recall 10pp。
