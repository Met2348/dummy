# L11 · 评测陷阱合集

## 1. Prompt 格式敏感

```
A.\nB.\nC.\nD.\n    vs    A) B) C) D)    vs    (A) (B) (C) (D)
```

同样模型，分数差 3-7pp。

**对策**：固定模板 + report 你用的 exact format。

## 2. Option 顺序偏差

模型有 "**positional bias**"：偏好 A 或 B（取决于训练数据）。

测试：把 4 个选项随机重排，跑 4 次取平均。

GPT-3.5 在 MMLU 上 A 偏好率 31%（应 25%）。

## 3. CoT 影响

```
Standard prompting:  pick A/B/C/D directly
CoT prompting:       "Let's think step by step. ... Therefore A."
```

CoT 在推理 task 上 +10-25pp，在知识 task 上几乎 0。

**报告时必须说**：是否用 CoT。

## 4. Answer Extraction Bug

模型生成 "I think A is most likely, but B is also possible."
- 严格 regex 取 "A" → 算对
- 严格取 "first letter" → 也是 A
- 但模型真意可能是 B

**对策**：
- 多种正则取并集，看一致性
- 用 LLM judge 抽答案
- report 抽取失败率

## 5. K-shot Demo 选择

不同 5-shot demo 组合，MMLU 分数差 5pp。
有的 demo 集"幸运" → 不可复现。

**对策**：固定 official demo set。

## 6. Tokenizer 边界

```
"Answer: A" → token_ids: [..., "Answer", ":", " A"]
"Answer:A"  → token_ids: [..., "Answer", ":A"]  ← 空格丢了
```

log-prob 评测特别敏感 — 漏空格直接 wrong。

## 7. Floating-point Numeracy

```
gold: "0.5"   pred: "1/2"   ← 都对，正则误判
gold: "0.50"  pred: "0.5"   ← 严格相等 fail
```

**对策**：math-verify、sympy parse、tolerance。

## 8. 数据集 Update Drift

```
2023.06 lm-eval-harness MMLU = HuggingFace `cais/mmlu`
2024.01 上游修了 label 错误
你 2023.12 跑的分数和 2024.02 跑的不可比
```

**对策**：lock data hash / commit。

## 9. Few-shot Demo 污染

5-shot demo 选自 train split → 没问题。
有人不小心选自 test split → 答案抄过去。

**对策**：严格隔离 split。

## 10. Eval 模式不一致

```
HF Leaderboard v1 用 log-prob
另一篇 paper 用 generative + regex
```

同 task 分数差 10pp，不可比。

**对策**：明确说 (log-prob / generative)，提供 raw outputs。

## 一句话

> 评测的 "1+1 = 2" 没那么简单。报告分数 + 报告**怎么算的**。
