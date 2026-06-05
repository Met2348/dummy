# L13 · RAGAS — RAG 评测框架（Es 2023）

## 30 秒核心

> RAGAS = RAG **A**ssessment：用 LLM-as-judge 给 RAG 输出打分。无需人工标注。

## 4 大指标

| 指标 | 测什么 | 谁判 |
|------|--------|------|
| **Faithfulness** | 答案是否被 context 支持 | LLM 判 (claim → context entail) |
| **Answer Relevancy** | 答案与 query 多相关 | LLM 反问 (从答案猜 query) |
| **Context Precision** | 检索 context 多相关 | LLM 判每块 relevant? |
| **Context Recall** | retrieve 是否找全 | 需 ground truth |

## Faithfulness 算法

```
1. LLM 把 answer 拆成 atomic claims
2. 每个 claim：LLM 判 "是否被 context 支持？" (是/否)
3. score = 支持的 claim 数 / 总 claim 数
```

## Answer Relevancy 算法

```
1. 用 answer 反向生成 N 个可能 query
2. 计算 cos(original_query, generated_queries) 平均
3. 高相关性 = 答案紧扣问题
```

## Context Precision 算法

```
1. 对每个 chunk: LLM 判 "对回答 query 是否相关？"
2. score = 加权平均（前排相关位置加权高）
```

## Context Recall 算法（需 GT 答案）

```
1. 把 GT answer 拆成 claims
2. 每个 GT claim：能从 context 推出？
3. score = 可推 claims / 总 claims
```

## Mock 实现 (`ragas_metrics.py` 预告)

```python
def faithfulness(answer, context):
    claims = split_claims(answer)
    supported = sum(1 for c in claims if any_substring_match(c, context))
    return supported / max(1, len(claims))

def answer_relevancy(answer, query, embed):
    return cos(embed(answer), embed(query))

def context_precision(query, contexts):
    rel = [keyword_overlap(query, c) for c in contexts]
    return sum(rel) / max(1, len(rel))

def context_recall(gt_answer, contexts):
    gt_claims = split_claims(gt_answer)
    recallable = sum(1 for g in gt_claims if any_substring_match(g, " ".join(contexts)))
    return recallable / max(1, len(gt_claims))
```

## 真实 RAGAS 库

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset,  # question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
```

需 LLM API（默认 GPT-4-turbo），$0.05-0.5 per 100 samples。

## 替代框架

| 工具 | 团队 |
|------|------|
| **RAGAS** | ExplodingGradients |
| **TruLens** | TruEra |
| **DeepEval** | Confident AI |
| **Promptfoo** | OSS |
| **Phoenix** | Arize |

## 退出条件

- 能默写 4 指标
- 能写 mock faithfulness
- 知道 RAGAS 是 LLM-as-judge

## 一句话

> RAGAS 4 维 (faithfulness / answer relevancy / context precision / context recall) —— LLM-as-judge 评 RAG，0 人工。
