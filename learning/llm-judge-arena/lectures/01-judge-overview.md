# L01 · LLM-as-Judge 4 类

## 为什么需要 judge

开放式生成（写作 / 对话 / 创意）**没 exact match**：
- "写一首关于秋天的诗" → 100 个正确答案
- exact match 完全失效

解决：让另一个 LLM 当裁判。

## 4 类 judge

| 类 | 输入 | 输出 | 例子 |
|----|------|------|------|
| **Pointwise** | (q, response) | score 1-10 | MT-Bench, G-Eval, Prometheus 2 |
| **Pairwise** | (q, resp_A, resp_B) | A / B / tie | Arena-Hard, AlpacaEval |
| **Listwise** | (q, [resp_1, ..., resp_N]) | ranking | ALC, custom |
| **Panel** | N judges 投票 | 多数 | JudgeBench, custom |

## 优缺点

| 类 | 优点 | 缺点 |
|----|------|------|
| Pointwise | 简单、便宜 | 分数漂移 |
| Pairwise | 相对稳定 | O(N²) pairs |
| Listwise | 一次多对 | LLM ctx 限制 |
| Panel | 抗 bias | 成本高 |

## 聚合方式

```
Pointwise:  mean(scores)
Pairwise:   win_rate(A vs B) → Bradley-Terry → Elo
Listwise:   Spearman rank correlation
Panel:      majority vote
```

## 关键公式 — Bradley-Terry

```
P(A beats B) = exp(r_A) / (exp(r_A) + exp(r_B))
```

从 pairwise wins 反推每个模型的 latent strength r。
Chatbot Arena 用 BT + bootstrap。

## 当代 judge 选择

| 场景 | 推荐 judge |
|------|-----------|
| 大规模研究 (1000+ pair) | GPT-4o / Claude 3.5 |
| 开源 / 本地 | Prometheus 2 / Skywork-Critic |
| 业务对齐 | fine-tuned domain judge |
| 安全敏感 | Llama Guard / WildGuard |

## 本 Topic 覆盖

L02-L05: 主流 bench (MT-Bench / Arena-Hard / Chatbot Arena / AlpacaEval)
L06-L07: 开源 judge (Prometheus 2 / G-Eval)
L08-L09: bias 与 JudgeBench
L10: pairwise vs pointwise 对照
L11: cost engineering
L12: Capstone mini-arena

## 一句话

> 当 exact match 失灵时，让 LLM 当裁判 — 但裁判也有 bias。
