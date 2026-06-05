# L09 · JudgeBench — 元评测 (评 judge)

**Tan et al. 2024** · arXiv 2410.12784

## 核心动机

判 LLM 用 LLM judge，但 judge 自己准不准？
→ 用 ground truth 判 judge 准确率。

## 数据

- **350 pair**（response A 真比 response B 好，有明确证据）
- 4 类：
  - Knowledge (factual correctness)
  - Reasoning (step quality)
  - Math (verifiable answer)
  - Coding (test pass)
- 评判 judge：**judge 选对 = +1**

## judge accuracy 排行

| Judge | JudgeBench acc |
|------|---------------|
| GPT-4o | 56.9% |
| Claude 3.5 Sonnet | 61.4% |
| GPT-4 | 53.4% |
| Llama 3.1 405B | 58.2% |
| **o1** | **75.4%** |
| **R1** | **80%** |
| 随机 baseline | 50% |

→ 即使 GPT-4 也只 57%，远低于"应该 100%"。

## 启示

1. **强 judge ≠ 强模型**：推理模型 judge 更准
2. **领域差异大**：math 几乎 100%，knowledge 50%
3. **pair 必须 ground truth**：才能判 judge 对错

## 与 RewardBench 区别

| Bench | 测什么 |
|-------|------|
| **JudgeBench** | LLM-as-Judge 对/错 |
| **RewardBench** (2024) | Reward Model 对/错 |
| **Chatbot Arena** | 人类偏好 |

JudgeBench 偏 reasoning，RewardBench 偏 chat preference。

## 工程含义

如果你的 judge 准确率 ~50%：
- pairwise 结果 = 随机
- 排行无意义

→ judge **验过准确率才能上**。

## 怎么测自己的 judge

```
1. 拿 JudgeBench 350 pair
2. 跑 judge，看 acc
3. 若 < 60% → 换 judge / 换 prompt
```

src 没专门 runner（数据集需 download）。

## 一句话

> Judge 也要被 judge — JudgeBench 让你知道自己的"裁判"是不是瞎子。
