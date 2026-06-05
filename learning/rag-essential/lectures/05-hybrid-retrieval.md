# L05 · Hybrid Retrieval（BM25 + Vector）⭐⭐⭐⭐

## 30 秒核心

> 单 vector retrieval 在**长尾术语 / 缩写 / 标识符**上漏检 → 加 BM25 (sparse) 兜底 → RRF 融合分数。

工业 RAG 几乎全部用 hybrid，naive 只用 vector 是教学版。

## BM25 (Robertson 1994)

经典词袋打分，2024 仍是 sparse retrieval 标准：

```
BM25(q,d) = Σ_{t in q} IDF(t) * (f(t,d)*(k+1)) / (f(t,d) + k*(1-b+b*|d|/avgdl))

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

参数：k≈1.5, b≈0.75
```

| 词 | f(t,d) | IDF(t) | 贡献 |
|----|-------|-------:|-----:|
| "the" | 大 | 小 | 小 |
| "Claude4" | 小 | 大 | 大 |

## Hybrid: RRF (Reciprocal Rank Fusion)

```
RRF_score(d) = Σ_{ranker r} 1 / (k + rank_r(d))
              k 默认 60
```

→ 不需要分数尺度对齐（vector cosine 与 BM25 分数尺度不同），只用 rank。

## 实现 (`hybrid.py` 预告)

```python
def rrf_fusion(vector_ranks: list[str], bm25_ranks: list[str], k=60):
    scores = {}
    for rank, doc in enumerate(vector_ranks, start=1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)
    for rank, doc in enumerate(bm25_ranks, start=1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

## Hybrid 数字（论文/工业实测）

| 设置 | recall@10 |
|------|----------:|
| Dense only (BGE) | 70% |
| BM25 only | 62% |
| Hybrid RRF | **78%** |

→ 1+1 > 2，且对"漏检"敏感场景（医疗、法律、代码搜索）尤其明显。

## 何时 hybrid 受益最大

| 场景 | 受益 |
|------|------|
| 代码搜索（函数名） | ⭐⭐⭐⭐⭐ |
| 法律（条款编号） | ⭐⭐⭐⭐⭐ |
| 医疗（药名缩写） | ⭐⭐⭐⭐ |
| 通用问答 | ⭐⭐⭐ |
| 多语言混排 | ⭐⭐⭐ |

## 工业实现

| 系统 | hybrid 支持 |
|------|------------|
| Weaviate | ✓ 内置 |
| Qdrant | ✓ sparse vector 支持 |
| Elasticsearch | ✓ + 8.x dense_vector |
| Pinecone | ✓ Sparse-Dense Index |
| Postgres pgvector | ✓ + ts_rank 或 BM25 extension |

## RRF 替代：linear / α-tuned

```
score = α * vector_cos + (1-α) * bm25_normalized
```

需 calibration set 调 α。RRF 不需 calibration，工业默认。

## 退出条件

- 能写 BM25 IDF 公式
- 能解释 RRF 不需分数尺度对齐
- 知道 hybrid recall@10 比 dense 高 8pp

## 一句话

> Hybrid = BM25 兜底术语 + dense vector 兜底语义 —— RRF 融合 rank，不调参，工业默认。
