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
| L14 | **Capstone**: 6 策略对照 | `capstone_rag_compare.py` |

## Tags

- `rag-essential` — Module 7 第 2 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（15/15，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules rag-essential
> ```

15 个脚本全是**手写 RAG 算法 + mock hash-embedding / mock reranker / mock LLM**（零外部依赖、纯 stdlib，见
`environment/requirements.txt`）。每个直跑都会执行内置 `_self_test()`（真断言，非 print-only）。直接
`python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/` 加进 `PYTHONPATH`，Python 本身也会把脚本所在
目录插入 `sys.path[0]`，故脱离 harness 单独跑也不依赖 CWD）：

```powershell
# 共享后端（Doc/Chunk/RetrievalResult/hash_embed/cosine + 20 篇样例文档）
python learning/rag-essential/src/common.py
# L03 Chunking 策略（fixed / sentence / semantic）
python learning/rag-essential/src/chunker.py
# L02 Naive RAG（embed + cosine top-k）
python learning/rag-essential/src/naive_rag.py
# L05 Hybrid retrieval（BM25 + dense + RRF fusion）
python learning/rag-essential/src/bm25_minimal.py
python learning/rag-essential/src/hybrid.py
# L06 Reranker（关键词重合度模拟 cross-encoder）
python learning/rag-essential/src/reranker_mock.py
# L07 ColBERT（per-token 向量 + MaxSim late interaction）
python learning/rag-essential/src/colbert_minimal.py
# L08 HyDE（mock LLM 写假设性答案再检索）
python learning/rag-essential/src/hyde_demo.py
# L09 GraphRAG（entity 抽取 + 连通分量社区 + 摘要）
python learning/rag-essential/src/graph_rag.py
# L10 HippoRAG（entity graph + personalized PageRank）
python learning/rag-essential/src/hipporag.py
# L11 RAG-Fusion / Multi-Query（query 改写 + RRF）
python learning/rag-essential/src/rag_fusion.py
# L12 Self-RAG/CRAG（confidence 三档分支 + mock web fallback；实现的是工程版 CRAG）
python learning/rag-essential/src/self_rag.py
# L13 RAGAS（faithfulness / answer_relevancy / context_precision / context_recall）
python learning/rag-essential/src/ragas_metrics.py
# paper 伴读：RAG-Sequence/RAG-Token 边缘化 + DPR 内积 + index hot-swap（Lewis 2020）
python learning/rag-essential/src/rag_original_minimal.py
```

**Capstone（L14）：6 策略（naive / hybrid / hybrid+rerank / HyDE / GraphRAG / HippoRAG）x 4 RAGAS 维度横评**

```powershell
python learning/rag-essential/src/capstone_rag_compare.py
```

> ⚠️ **hash-embedding 非固定种子坑**：mock embedding（`common.hash_embed`）用 Python 内置 `hash()` 分桶，
> 该函数**逐进程随机加盐**（`PYTHONHASHSEED` 未固定），所以 capstone 打印的具体分数**每次运行会有小幅浮动**
> （抽样 25 次独立进程重跑实测：排名结论稳定——GraphRAG 稳定夺冠，无一次 FAIL），这是预期行为不是 bug；
> `_self_test()` 全用相对/阈值断言（如"更相似的文本 cosine 更高""mean > 0.2"），不依赖具体数值，故不受影响。

**测试（V2）**：

```powershell
python learning/rag-essential/src/tests/test_rag.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules rag-essential --tests
```

预期：`15/15 modules passed`。（`test_rag.py` 是脚本式聚合跑手，无 pytest `test_` 函数；pytest 收集会得
0 项，`--tests` harness 侦测到后自动回退为直跑该脚本，断言真实执行。）

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

> 6 RAG 策略 + RAGAS 4 维评测 — 全部 stdlib mock，理解原理而非依赖框架。
