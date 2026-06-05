# L06 · Prometheus 2 — 开源 judge

**Kim et al. 2024** · arXiv 2405.01535 · KAIST

## 背景

GPT-4 judge **太贵**：1 题 ~$0.05，100k pair ~$5000+。
社区急需**开源 judge 模型**。

## Prometheus 2 数据

- **100k pointwise + 100k pairwise** GPT-4 标注样本
- 训练：Mistral 7B / Mixtral 8x7B base
- 输出：1-5 score + reasoning

## Rubric-based

每题带 5-point rubric：

```
Score 1: Response shows no understanding of the question.
Score 2: Response touches on the topic but lacks depth.
Score 3: Response addresses the question but has minor inaccuracies.
Score 4: Response is accurate and well-structured.
Score 5: Response is comprehensive and insightful.
```

模型按 rubric 打分。

## 性能

| Judge | Pearson w/ GPT-4 (Vicuna-bench) |
|-------|--------------------------------|
| Llama-2 13B (no FT) | 0.31 |
| GPT-3.5 | 0.52 |
| Prometheus 1 (Llama-2 13B) | 0.59 |
| **Prometheus 2 7B** | **0.78** |
| **Prometheus 2 8x7B** | **0.85** |
| GPT-4 (self) | 1.0 |

Prometheus 2 8x7B **接近 GPT-4** 评分一致性。

## 相关 open judge

| Judge | 大小 | Pearson |
|-------|------|--------|
| Prometheus 2 | 7B/56B | 0.85 |
| **Skywork-Critic** (2024) | 8B | 0.87 |
| JudgeLM (2023) | 7B/33B | 0.71 |
| PandaLM (2023) | 7B | 0.65 |

## 实操

src/prometheus2_judge.py 用 keyword-matching mock 演示 rubric 模式：

```python
from prometheus2_judge import score_rubric, COMMON_RUBRICS

r = COMMON_RUBRICS["helpfulness"]
score_rubric("step by step. Specifically, example...", r)  # 5
```

## 部署

```python
from vllm import LLM, SamplingParams
prom2 = LLM("prometheus-eval/prometheus-2-7b")
# input: rubric + response → output score 1-5
```

5090 24GB 跑 7B 快，跑 56B MoE 紧。

## 一句话

> Prometheus 2 = 开源版 GPT-4 judge，省钱 + 可控。
