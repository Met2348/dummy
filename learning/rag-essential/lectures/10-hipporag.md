# L10 · HippoRAG（Gutiérrez 2024, OSU）

## 30 秒核心

> 灵感：海马体（hippocampus）记忆机制。
> **PageRank** in entity graph + dense retrieval → 比 GraphRAG 便宜 6-10×、快 6-13×、精度相当。

NeurIPS 2024 spotlight。

## 名字由来

```
Hippocampus = 大脑海马体，负责 episodic memory + relational reasoning
              "哪两个事实可以连起来回答这个新问题？"

HippoRAG    = 模仿这种"联想" — 用图 + PageRank 找跨文档关联
```

## 流程

```
Index:
  1. Doc → triples (LLM-OpenIE)
  2. 建图，节点 = entity，edge = co-occurrence + relation
  3. 索引 entity passage (entity ↔ 出现的 doc)

Query:
  1. 抽 query entity
  2. Personalized PageRank（PPR），起点 = query entity
  3. 高 PPR 分的 entity → 对应 doc → top-k
```

## 与 GraphRAG 区别

| 维度 | GraphRAG | HippoRAG |
|------|----------|----------|
| Community | Louvain → summary | 无 community，全图 PPR |
| Summary 成本 | 每 community LLM call | 无 |
| Index 成本 | 高 | 中 |
| Query 成本 | 中 | 低 |
| 适用 | 全局/总体问题 | multi-hop QA |

## Personalized PageRank（PPR）

```
标准 PageRank: 所有节点初始权重均匀
Personalized:  从 query entity 出发，跳转 prob 偏向起点

PPR(v) = α * I(v == start) + (1-α) * Σ_{u → v} PPR(u) / out_degree(u)
α ≈ 0.15
```

→ "从 query entity 出发，扩散到相关 entity"。

## Multi-hop QA 数字（MuSiQue, 2WikiMultiHopQA）

| Method | F1 |
|--------|----:|
| Naive RAG | 33% |
| GraphRAG | 51% |
| **HippoRAG** | **57%** |
| HippoRAG + IRCoT | **63%** |

## 实现 (`hipporag.py` 预告)

```python
class HippoRAG:
    def index(self, docs):
        for d in docs:
            ents = extract_entities(d)
            for e in ents:
                self.entity_to_doc[e].add(d)
            for e1 in ents:
                for e2 in ents:
                    if e1 != e2:
                        self.graph[e1][e2] += 1  # co-occur

    def query(self, q, k=5):
        q_ents = extract_entities(q)
        ppr = personalized_pagerank(self.graph, q_ents, alpha=0.15)
        ranked_ents = sorted(ppr, key=ppr.get, reverse=True)
        docs = []
        for e in ranked_ents:
            docs.extend(self.entity_to_doc[e])
            if len(docs) >= k: break
        return docs[:k]
```

## HippoRAG 2 (Gutiérrez 2025.02)

- 加 sub-graph 修剪
- 加 IRCoT (iterative retrieval CoT)
- MuSiQue F1 70%+

## 退出条件

- 能讲海马体类比
- 能默写 PPR 公式
- 知道与 GraphRAG 数字对照

## 一句话

> HippoRAG = 海马体类比 + PPR —— 比 GraphRAG 便宜 6-10× 精度相当，multi-hop QA 强。
