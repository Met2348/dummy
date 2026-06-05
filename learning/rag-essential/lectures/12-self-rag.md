# L12 · Self-RAG / CRAG

## Self-RAG（Asai 2023）

LLM 自己判断**何时 retrieve、retrieve 是否相关、答案是否 supported**。

### 4 个特殊 token

| Token | 意义 |
|-------|------|
| `[Retrieve]` | 需要检索 |
| `[Relevant]` | 检索结果相关 |
| `[Supported]` | 答案被 retrieve 支持 |
| `[Useful]` | 答案 useful |

### 流程

```
Q → LLM 输出 [Retrieve] → retrieve docs
   → 每个 doc：LLM 判 [Relevant?] → 留下 relevant
   → 生成 → LLM 判 [Supported?] [Useful?]
```

LLM 是 fine-tune 过的，会自然输出这些 token。

## CRAG（Corrective RAG, Yan 2024）

Self-RAG 的"工程版"，不需 fine-tune：

```
1. retrieve
2. 评估检索质量（confidence 高/中/低）
3. confidence 高 → 直接用
   confidence 中 → 多 query + 网络搜索补充
   confidence 低 → 完全转网络搜索
4. generate
```

## 实现 (`self_rag.py` 预告)

```python
def crag(q, retriever, web_search, llm):
    docs = retriever(q)
    confidence = evaluator(q, docs)  # mock: keyword overlap
    if confidence > 0.7:
        return llm(f"Q:{q}\nCtx:{docs}\nA:")
    elif confidence > 0.3:
        more = web_search(q)
        return llm(f"Q:{q}\nCtx:{docs + more}\nA:")
    else:
        web = web_search(q)
        return llm(f"Q:{q}\nCtx:{web}\nA:")
```

## RAG 质量评估的 4 个 dimension

| 维 | 指标 |
|----|------|
| 检索准确性 | recall@k / nDCG |
| 答案 faithfulness | RAGAS faithfulness |
| 答案 relevancy | RAGAS answer_relevancy |
| Context precision | RAGAS context_precision |

L13 详讲 RAGAS。

## 何时不用 Self-RAG / CRAG

| 场景 | 跳过 |
|------|------|
| 任务窄、retrieve 总好 | 浪费 token |
| Latency 严格 | 多 LLM 判 token 慢 |
| 索引明显高质 | direct retrieval 够 |

## 退出条件

- 能讲 Self-RAG 4 token
- 能讲 CRAG 三 confidence 分支
- 知道 CRAG 不需 fine-tune

## 一句话

> Self-RAG = 4 token 自评 (retrieve/relevant/supported/useful)；CRAG = 三档 confidence 分支 — 都让 RAG 学会"自我反思"。
