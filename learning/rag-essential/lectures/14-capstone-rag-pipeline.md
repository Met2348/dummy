# L14 · Capstone — 5 策略 RAG 横评 ⭐

## 任务

> 50 mock 文档 + 20 mock query，跑 5 RAG 策略，RAGAS 4 维评分对照。

## 5 策略

| # | 策略 | 实现 |
|---|------|------|
| 1 | naive | `naive_rag` (cosine top-k) |
| 2 | hybrid | BM25 + dense + RRF |
| 3 | hybrid + rerank | + keyword_overlap_rerank |
| 4 | HyDE | mock hypo + retrieve |
| 5 | GraphRAG mock | entity graph + local query |

## 输出表

| Strategy | Faithfulness | Answer-Rel | Ctx-Prec | Ctx-Recall | Mean |
|----------|--------------|-----------|---------|-----------|------|
| naive    | 0.60 | 0.65 | 0.55 | 0.50 | 0.58 |
| hybrid   | 0.70 | 0.70 | 0.68 | 0.65 | 0.68 |
| h+rerank | 0.78 | 0.75 | 0.80 | 0.70 | 0.76 |
| HyDE     | 0.72 | 0.73 | 0.65 | 0.68 | 0.70 |
| GraphRAG | 0.82 | 0.74 | 0.75 | 0.78 | 0.77 |

→ GraphRAG / hybrid+rerank 并列冠军（不同 use case 偏好不同）

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/rag-essential/src'); from capstone_rag_compare import run_compare, to_md; print(to_md(run_compare()))"
```

## 退出条件

- 5 策略全跑通
- 4 维 score 输出 markdown 表
- 决策树写明"哪 use case 选谁"

## 一句话

> 50 doc × 20 query × 5 strategy × 4 RAGAS = 1 张 selection 决策表。
