# L07 · ColBERT — Late Interaction（Khattab 2020）

## 30 秒核心

> ColBERT = 每个 token 独立 embedding，**MaxSim** 聚合 → 兼具 dual-encoder 速度和 cross-encoder 精度。

ICLR 2020，引用 2000+，事实上的 hybrid 第三选项。

## 三种 retrieval 体系

```
1. Dual encoder
   q → 1 vector
   d → 1 vector
   score = cos
   ✓ 快  ✗ 信号弱

2. Cross encoder (reranker)
   (q,d) → 1 vector
   score = linear
   ✗ 慢  ✓ 精度高

3. ColBERT (late interaction)
   q → [v_q1, v_q2, ...]   (per-token)
   d → [v_d1, v_d2, ...]
   score = Σ_i max_j cos(v_qi, v_dj)
   ✓ 中速  ✓ 中精
```

## MaxSim 公式

```
ColBERT(q, d) = Σ_{i in q} max_{j in d} cos(E_q[i], E_d[j])
```

**直观**：每个 query token 找文档里最像它的 token，加起来。

## 实现 (`colbert_minimal.py` 预告)

```python
def colbert_score(q_tokens, d_tokens, embed):
    q_vecs = [embed(t) for t in q_tokens]
    d_vecs = [embed(t) for t in d_tokens]
    s = 0.0
    for qv in q_vecs:
        s += max(cos(qv, dv) for dv in d_vecs)
    return s
```

## 为什么 work

| 现象 | 解释 |
|------|------|
| Long query 表现好 | 每 token 独立"找配" |
| 长尾词不丢 | 罕见 query token 也能 max |
| 跨长文档 | doc 长不平均压缩信息 |

## ColBERTv2 (Santhanam 2022)

- 残差压缩：vector 量化 → 存储减 6 倍
- 工业可用：Vespa / Pyserini 集成

## RAGatouille (Bavanari 2024)

ColBERT 的 Pythonic 包装：
```python
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
RAG.index(docs, ...)
RAG.search(query, k=5)
```

→ "想用 ColBERT 但不想训练" 的标准路径。

## ColBERT vs dense + reranker

| 维度 | ColBERT | dense + rerank |
|------|---------|----------------|
| 离线索引 | per-token vector（大 5-10×） | 1 vector / doc |
| 在线 search | MaxSim 全 token | 2 阶段 |
| 精度 | 高（MTEB 65-68） | 高（68-72） |
| 存储 | 大 | 小 |

## 退出条件

- 能默写 MaxSim 公式
- 能解释 ColBERT 介于 dual / cross 之间的位置
- 知道 RAGatouille 是 ColBERT 工程包装

## 一句话

> ColBERT = 每个 token 独立 embed + MaxSim 聚合 —— dual 的速度 × cross 的精度，per-token 存储是代价。
