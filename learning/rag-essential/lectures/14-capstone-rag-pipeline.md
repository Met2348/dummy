# L14 · Capstone — 6 策略 RAG 横评 ⭐

## 任务

> 20 mock 文档 + 10 mock query（`common.SAMPLE_DOCS` / `SAMPLE_QUERIES`），跑 6 RAG 策略，RAGAS 4 维评分对照。

## 6 策略

| # | 策略 | 实现 |
|---|------|------|
| 1 | naive | `naive_rag` (cosine top-k) |
| 2 | hybrid | BM25 + dense + RRF |
| 3 | hybrid + rerank | + keyword_overlap_rerank |
| 4 | HyDE | mock hypo + retrieve |
| 5 | GraphRAG mock | entity graph + local query |
| 6 | HippoRAG | entity graph + personalized PageRank |

## 输出表（示例，某次实测；具体分数逐次浮动见下方坑注记）

| Strategy | Faithfulness | Answer-Rel | Ctx-Prec | Ctx-Recall | Mean |
|----------|--------------|-----------|---------|-----------|------|
| naive         | 1.00 | 0.42 | 0.13 | 0.70 | 0.56 |
| hybrid        | 1.00 | 0.39 | 0.23 | 0.70 | 0.58 |
| hybrid+rerank | 1.00 | 0.39 | 0.24 | 0.70 | 0.58 |
| HyDE          | 1.00 | 0.27 | 0.29 | 0.90 | 0.61 |
| GraphRAG      | 1.00 | 0.37 | 0.44 | 0.80 | **0.65** |
| HippoRAG      | 0.90 | 0.24 | 0.17 | 0.40 | 0.43 |

→ GraphRAG 通常夺冠（entity 图 + community 摘要对这批样例文档天然占优）。

> ⚠️ **坑**：`Answer-Rel`/`Ctx-Prec` 等列用 `common.hash_embed`（Python 内置 `hash()` 分桶，逐进程随机加盐，
> `PYTHONHASHSEED` 未固定），所以每次重跑具体分数会有 ±0.05~0.1 左右浮动——这是预期行为（`_self_test` 只做
> 相对/阈值断言），不是复现失败；关注的应是**策略间相对高低**和**排名结论**，不是小数点后第 3 位。

## 跑

```powershell
python learning/rag-essential/src/capstone_rag_compare.py
```

## 退出条件

- 6 策略全跑通
- 4 维 score 输出 markdown 表
- 决策树写明"哪 use case 选谁"

## 一句话

> 20 doc × 10 query × 6 strategy × 4 RAGAS = 1 张 selection 决策表。
