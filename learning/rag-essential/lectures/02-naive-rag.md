# L02 · Naive RAG

## 5 步流程

```
1. Load:  原文档（txt/pdf/md/html）
2. Chunk: 切块（500-1000 token）
3. Embed: 每块 → vector
4. Search: query → vector → top-k 块
5. Generate: 块 + query → LLM
```

## 最小实现

```python
def naive_rag(query, docs, embed, llm, k=3):
    chunks = [c for d in docs for c in chunk(d)]
    chunk_vecs = [embed(c) for c in chunks]
    q_vec = embed(query)
    scored = sorted(zip(chunk_vecs, chunks),
                    key=lambda cv: cosine(q_vec, cv[0]),
                    reverse=True)
    top = [c for _, c in scored[:k]]
    prompt = "Context:\n" + "\n".join(top) + f"\n\nQ: {query}\nA:"
    return llm(prompt)
```

## 4 大问题

| 问题 | 表现 | 后续 lecture 解 |
|------|------|----------------|
| Chunk 切歪 | 答案断 | L03 chunking |
| Embed 不准 | 关键词找不到 | L05 hybrid (BM25) |
| Top-k 噪声 | 无关块挤进来 | L06 reranker |
| 同义词/缩写 | 漏 retrieve | L08 HyDE |

## Mock embedding 实现（教学用）

```python
def hash_embed(text, dim=64):
    vec = [0.0] * dim
    for token in text.lower().split():
        vec[hash(token) % dim] += 1.0
    norm = sum(v*v for v in vec) ** 0.5
    return [v/(norm+1e-9) for v in vec]
```

bag-of-words hash 投影，不需 torch/transformers，足够演示 cosine retrieval。

## Cosine vs dot product vs Euclidean

| 度量 | 公式 | 何时用 |
|------|------|--------|
| Cosine | a·b / ‖a‖‖b‖ | 通用 |
| Dot | a·b | 已归一化时（OpenAI embedding） |
| Euclidean | ‖a-b‖ | 几何意义 |
| L2 vs IP | (norm 后等价) | FAISS index 选 |

## 退出条件

- 能背 5 步流程
- 能列 4 大问题
- 会写 cosine 函数

## 一句话

> Naive RAG = chunk + embed + cosine top-k + LLM —— 一切复杂 RAG 的"零号实现"。
