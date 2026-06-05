# L08 · HyDE — Hypothetical Document Embedding（Gao 2022）

## 30 秒核心

> Query 短而抽象，doc 长而具体 → embedding 空间不对齐。
> **HyDE**：让 LLM 先编一个"假想答案"，用假答案的 embedding 检索。

ACL 2023，零样本 retrieval 增强。

## 流程

```
Query: "What's the population of Tokyo?"
   ↓
[LLM] "Tokyo has 13.96 million people as of 2023..."
   ↓ (hypothetical answer)
embed(hypo) → search → top docs
```

## 为什么 work

| 失败模式 | HyDE 解 |
|----------|---------|
| Query 用术语，doc 用大白话 | 假答案与 doc 风格匹配 |
| Query 简短，embedding 信号稀 | 假答案密集 |
| Query 抽象，doc 具体数字 | 假答案带具体数字 |

→ "用预期答案搜，而不是用问题搜"。

## 风险

| 风险 | 缓解 |
|------|------|
| LLM 编错（hallucination） | 不要紧 — 我们只取 embedding，不读内容 |
| LLM 编得太具体（误导）| 用 temperature 高，多 hypothesis ensemble |
| 多调一次 LLM 慢 | 短 hypo 即可 (100 tok) |

## 实现 (`hyde_demo.py` 预告)

```python
def hyde_retrieve(query, llm, embed, doc_vecs, docs, k=5):
    hypo = llm(f"Write a passage answering: {query}")
    h_vec = embed(hypo)
    scored = sorted(zip(doc_vecs, docs),
                    key=lambda dv: cos(h_vec, dv[0]),
                    reverse=True)
    return [d for _, d in scored[:k]]
```

## 数字（Gao 2022）

| Setting | nDCG@10 (BEIR avg) |
|---------|-------------------:|
| Contriever (baseline) | 32.0 |
| HyDE + Contriever | **39.0** |

零样本 +7pp，且不需 fine-tune retriever。

## HyDE 变体

| 变体 | 改进 |
|------|------|
| Multi-HyDE | 生成 5 假答案 → avg embedding |
| Step-back HyDE | 先 generalize query 再 hypothesize |
| Query2Doc (微软) | 类似 HyDE 思想，2023 |

## 退出条件

- 能讲 HyDE 三句话
- 知道 +7pp 数字
- 能写 hyde_retrieve

## 一句话

> HyDE = 让 LLM 编个"标准答案"再用它检索 —— 用预期答案搜，零样本提 BEIR 7pp。
