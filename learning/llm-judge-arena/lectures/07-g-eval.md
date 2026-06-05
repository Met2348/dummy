# L07 · G-Eval — NLG 评分 with CoT

**Liu et al. 2023** · arXiv 2303.16634 · Microsoft + UCSB

## 核心 idea

让 LLM 当 NLG 评分员：
1. 给 task definition + criteria
2. 让 LLM **CoT 推理** "这个 response 多少分"
3. 用 logits 取 score expectation（而不是字面输出）

## 步骤

```
Step 1: Auto-generate eval steps from criteria
Step 2: Provide source + summary + eval steps
Step 3: LLM thinks step-by-step
Step 4: LLM outputs final score
Step 5: Use form-filling probability weighting
```

## 关键技巧：form-filling

不要让 LLM 直接 "say 3"。而是：

```
prompt = "Score (1-5): ___"
logits = LLM(prompt)
P(token = "1"), P(token = "2"), ..., P(token = "5")
score = sum(P(i) * i for i in 1..5)
```

→ 平滑 score（比 argmax 稳）。

## NLG 评测维度

经典 4 维：
- **Coherence**: 是否连贯
- **Consistency**: 与 source 是否一致
- **Fluency**: 语言质量
- **Relevance**: 是否相关

## 性能（vs 人类）

| Metric | Pearson w/ human |
|--------|-----------------|
| BLEU | 0.17 |
| ROUGE | 0.21 |
| BERTScore | 0.23 |
| GPTScore | 0.50 |
| **G-Eval (GPT-4)** | **0.51-0.60** |

→ 是 NLG 评测的事实标准（HF Datasets 默认）。

## 与 Prometheus 区别

| 维度 | G-Eval | Prometheus 2 |
|------|--------|-------------|
| 推理 | CoT | CoT |
| 输出 | weighted score | direct score + reason |
| 训练 | prompt only (zero-shot) | 200k 训练样本 |
| 模型 | GPT-4 | open 7B / 56B |

## 已知 bias

G-Eval 论文揭示：**LLM judge 偏好 LLM-generated text** over human-written。
即使有 CoT 也消除不掉。

## 一句话

> G-Eval = 让 LLM 用 CoT + form-filling 当 NLG 评分员，HF 默认 metric。
