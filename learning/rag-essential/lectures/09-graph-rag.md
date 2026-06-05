# L09 · GraphRAG（Microsoft 2024）⭐⭐⭐⭐⭐

## 30 秒核心

> GraphRAG = **图结构索引** + **community summary** + **map-reduce 检索**。
>
> 解决问题：跨文档关系 ("X 和 Y 有什么联系？") + 全局问题 ("总体上 ...") naive RAG 找不到答案。

Microsoft Research 2024.04 paper + 开源 repo，2024 RAG 最大突破。

## Why naive RAG 答不了 "总体" 问题

```
Q: "这份 200 页报告的总体主题是什么？"
Naive RAG: top-5 chunks → 都是细节，没人总结全貌
GraphRAG: 已预生成 community summary → 直接答
```

## 流程

```
Index 阶段（离线）:
  1. LLM 抽 entity + relation → triple
  2. 建 knowledge graph
  3. Louvain / Leiden 算法找 community
  4. LLM 给每个 community 写 summary
  5. Hierarchical summary（层 0 元素 → 层 1 组 → 层 2 大组）

Query 阶段（在线）:
  Local query: entity-centric → 局部子图 + 相关 community summary
  Global query: map (每 community 答一段) → reduce (整合)
```

## Entity extraction

```python
prompt = f"""From: {chunk}
Extract: (entity1, relation, entity2) triples."""
triples = llm(prompt)
# → [(Claude, made_by, Anthropic), (Anthropic, founded_in, 2021), ...]
```

## Community detection

- Louvain（Blondel 2008）modularity 最大化
- Leiden（Traag 2019）改进 Louvain
- 输出：cluster id per node

## Community summary

```
对 community {Claude, Anthropic, Dario_Amodei, ...}:
LLM("Summarize this community of entities and their relations: ...")
→ "Anthropic 是 Claude 的开发者，由 Dario Amodei 创立 ..."
```

## Local vs Global query

| 类型 | 例 | 走哪个路径 |
|------|---|-----------|
| Local | "Claude 哪年发布？" | entity → community → answer |
| Global | "这份报告讲了什么？" | 全 community map-reduce |
| Drift | "X 与 Y 有什么不同？" | 多 community 交叉 |

## 实现 (`graph_rag.py` 预告)

```python
class GraphRAG:
    def index(self, docs):
        for d in docs:
            for triple in extract_triples(d):
                self.graph.add_edge(*triple)
        self.communities = louvain(self.graph)
        for c in self.communities:
            self.summaries[c] = summarize_community(c)

    def query_local(self, q):
        ents = extract_entities(q)
        comm = [self.community_of[e] for e in ents]
        return llm(f"Q:{q}\nCtx: {[self.summaries[c] for c in comm]}")

    def query_global(self, q):
        partials = [llm(f"Q:{q}\nCommunity: {s}") for s in self.summaries.values()]
        return llm(f"Q:{q}\nPartial answers: {partials}\nSynthesize:")
```

## 成本警告

- Index 阶段：每 chunk 调 LLM 抽 entity + 每 community 调 LLM 总结
- 100 页报告：~$5-50 一次性 index 成本
- 一旦 indexed，查询便宜

## 后续家族

| 系统 | 改进 |
|------|------|
| **LightRAG** (Guo 2024) | 简化 + 增量 update |
| **nano-graphrag** | 1k 行 Python 教学版 |
| **GraphRAG-Local-UI** | Ollama 本地化 |
| **HippoRAG** (L10) | 用 PageRank 代替 community |

## 退出条件

- 能讲 5 步 index + 2 query 路径
- 知道 GraphRAG 解决"总体问题"
- 知道 LightRAG / nano-graphrag 是简化版

## 一句话

> GraphRAG = 文档 → entity 图 + community summary —— 答得了"总体上 X" 的问题，但 index 成本 100×。
