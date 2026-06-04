# L01 · 推理 benchmark 全景

## 为什么推理是 2024-2026 主战场

- 知识 bench (MMLU) → 已饱和
- 对话 bench (MT-Bench) → judge bias 严重
- 推理 bench (GSM8K/MATH/AIME) → **仍在卷**，o1/R1 时代主指标

## 4 大类

| 类 | 代表 | 测什么 |
|----|------|------|
| **数学** | GSM8K / MATH / AIME | 多步计算 + 符号 |
| **科学** | GPQA / HLE | 专业知识 + 推断 |
| **逻辑** | ZebraLogic / BBH-logic | 约束满足 |
| **常识** | StrategyQA / OpenBookQA | 隐性知识 |

## 难度阶梯（2026 视角）

```
GSM8K (grade school)      → top: 95%+, 老 bench
MATH (competition)        → top: 90%+, 中端
AIME 2024 (national)      → top: 80%+, R1 战场
GPQA Diamond (PhD)        → top: 70%+
HLE (Humanity's Last Exam)→ top: 15%, 新王
ARC-AGI (Chollet)         → top: 50%+ (o3 解了), 真智力测试
```

## 评测协议要点

1. **多 sample**：温度 > 0，取 pass@1 / pass@k
2. **answer extraction**：严格 (math-verify) 或宽松 (regex)
3. **CoT** prompt 必备（rather than direct answer）
4. **去污染**：AIME 必须用未公开年份

## R1 时代的 metric

- **pass@1**: greedy 一次对
- **maj@N**: N 次投票
- **best-of-N**: PRM 排序选最佳

R1 paper 主报 `cons@64` (consensus@64 samples)。

## 本 Topic 覆盖

L02-L05: 数学 4 大 bench (GSM8K/MATH/AIME/HLE) + verifier
L06-L07: 科学 (GPQA) + HLE
L08-L09: 逻辑 + 工具增强
L10: 多轮数学推理
L11: 陷阱
L12: Capstone

## 一句话

> 推理 bench = 当今 LLM 的 IQ 测试，难度还在上升。
