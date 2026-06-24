# Topic 2: RAG Essential（检索增强全谱）

> Module 7「Agent 应用层」第 2 专题 · 14 lectures · ~14h
>
> 从 naive RAG → hybrid → reranker → ColBERT → HyDE → GraphRAG → HippoRAG → RAGAS

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | RAG 是什么 | (intro) |
| L02 | Naive RAG | `naive_rag.py` |
| L03 | Chunking 策略 | `chunker.py` |
| L04 | Embedding 模型 (text-embed-3 / BGE / Voyage) | (lecture) |
| L05 | **Hybrid retrieval (BM25+vector)** | `bm25_minimal.py` + `hybrid.py` |
| L06 | **Reranker** | `reranker_mock.py` |
| L07 | **ColBERT** late interaction | `colbert_minimal.py` |
| L08 | **HyDE** | `hyde_demo.py` |
| L09 | **GraphRAG** (Microsoft 2024) | `graph_rag.py` |
| L10 | **HippoRAG** (OSU 2024) | `hipporag.py` |
| L11 | RAG-Fusion / Multi-Query | `rag_fusion.py` |
| L12 | Self-RAG / CRAG | `self_rag.py` |
| L13 | **RAGAS** evaluation | `ragas_metrics.py` |
| L14 | **Capstone**: 5 策略对照 | `capstone_rag_compare.py` |

## Tags

- `rag-essential` — Module 7 第 2 专题

## 跑测试

```powershell
python learning/rag-essential/src/tests/test_rag.py
```

预期：`all modules passed`。

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/rag-essential/src'); from capstone_rag_compare import run_compare, to_md; print(to_md(run_compare()))"
```

## 5 策略选型决策树

| 场景 | 推荐 |
|------|------|
| Quick PoC | naive |
| 长尾术语 | hybrid (BM25+vector) |
| 高 faithfulness 要求 | hybrid + reranker |
| 跨文档关系 | GraphRAG |
| 历次 query 复用 | HippoRAG |

## 关键文献

- Naive RAG: Lewis 2020
- BM25: Robertson 1994
- ColBERT: Khattab 2020
- HyDE: Gao 2022
- GraphRAG: Edge 2024 (Microsoft)
- HippoRAG: Gutiérrez 2024 (OSU)
- RAGAS: Es 2023
- Self-RAG: Asai 2023
- CRAG: Yan 2024

## 一句话

> 5 RAG 策略 + RAGAS 4 维评测 — 全部 stdlib mock，理解原理而非依赖框架。


---
## 🔬 小而真 · 真实模型例子
> 除 toy 外, 本专题附一个**真实小模型** notebook (本地 gpt2/TinyLlama, CPU 离线):
> - [`notebooks/N15-real-rag.ipynb`](notebooks/N15-real-rag.ipynb) — 真实 RAG: gpt2 嵌入检索 + TinyLlama 接地生成 (闭卷瞎编 → 开卷有据)
> 共享工具见 [`learning/_shared/realmodels.py`](../_shared/realmodels.py)。
