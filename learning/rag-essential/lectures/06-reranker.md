# L06 · Reranker（精排）⭐⭐⭐⭐⭐

## 30 秒核心

> Retrieve 阶段拿 top-50 候选 → reranker 精排 → 留 top-5 给 LLM。
>
> Reranker 是 cross-encoder：同时看 (query, doc) → 比 dual-encoder embedding 准但慢。

## Dual vs Cross encoder

```
Dual (embedding):
  q  → encoder → v_q
  d  → encoder → v_d
  score = cos(v_q, v_d)
  → 离线索引 ✓ 快

Cross (reranker):
  (q, d) → encoder → score
  → 不能预索引 ✗ 慢
  → 但有交互层，精度高
```

## Reranker 主流型号

| 模型 | 团队 | 大小 | 用 |
|------|------|------|----|
| **rerank-3** | Cohere | API | 商业默认 |
| **bge-reranker-v2-m3** | BAAI | 568M | 开源默认 |
| **mxbai-rerank-large-v1** | Mixedbread | 435M | 新秀 |
| **Voyage rerank-2** | Voyage | API | Anthropic 推 |
| **ms-marco-MiniLM-L-12-v2** | sbert | 33M | 极速 |

## 工业 pipeline

```
Query → vector search (top 50)
      → BM25 (top 50)
      → RRF (top 50)
      → Reranker (top 5)
      → LLM
```

## 数字对照（Cohere blog 2024）

| 设置 | nDCG@10 |
|------|--------:|
| BM25 only | 0.45 |
| + dense | 0.55 |
| + rerank-3 | **0.68** |

→ Reranker 让 top-5 精度跳一档。

## Mock reranker（教学）

```python
def keyword_overlap_rerank(query, candidates, k=5):
    q_tokens = set(query.lower().split())
    scored = [(c, len(q_tokens & set(c.lower().split())) / max(1, len(q_tokens))) for c in candidates]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
```

教学用 token overlap 模拟 cross-encoder 倾向（与真 cross-encoder 学到的 attention 不同，但走通流程）。

## Reranker 何时不用

| 场景 | 跳过 reranker |
|------|--------------|
| Latency < 200ms 硬要求 | rerank +50-200ms 太贵 |
| 候选已 5-10 个 | 收益小 |
| LLM 自己当 reranker | "list-wise" prompt 也行 |

## LLM-as-reranker

```
"Rate each document 1-10 for relevance to query: ...
Docs: 1) ... 2) ... 3) ..."
```

→ GPT-4 / Claude 当 reranker，准但 token 贵。LangChain `LLMReranker` 即此。

## RankGPT / RankZephyr

- RankGPT (Sun 2023)：list-wise prompt
- RankZephyr (Pradeep 2023)：7B 模型 fine-tune 当 reranker

## 退出条件

- 能讲 dual vs cross encoder
- 能列 3 商业 + 2 开源 reranker
- 知道 nDCG@10 提升 ~10pp

## 一句话

> Reranker = 排队进 LLM 之前的"安检 + 优先级排队" —— Cohere rerank-3 / BGE-reranker-v2-m3 工业首选。
