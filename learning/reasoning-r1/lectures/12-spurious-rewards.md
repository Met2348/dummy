# L12 · Spurious Rewards — 2025.06 的"鬼故事"

> 16 slides | 45 min | R1 时代最重要的警示

---

## Slide 1 · 这个故事

2025.06 论文 *Spurious Rewards Make Reward Models Worse*:
> Qwen-2.5-7B 用 **随机奖励** GRPO 训练，MATH 涨 21pp.

行业震惊。

---

## Slide 2 · 设置

```
base: Qwen2.5-7B
algo: GRPO
reward: random ∈ {0, 1} （与回答无关）
data: MATH 训练集
```

理论上：信号完全无意义。

---

## Slide 3 · 结果

```
                MATH 测试 accuracy
base Qwen-7B:    45%
+ 真正 verifier: 65% (+20)
+ 随机奖励:      66% (+21) ⚠️
```

**随机奖励涨幅 ≈ 真奖励**。

---

## Slide 4 · 第一反应：bug?

不是 bug。多个独立团队重复出来：
- Stanford
- Berkeley
- Anthropic
- 自家 DeepSeek

→ 现象稳定。

---

## Slide 5 · 解释 #1: 形态学习

GRPO 会鼓励"格式化输出"：
- 即使 reward 随机，model 学会"用 CoT"
- CoT 本身就提升 accuracy
- → reward 只是"激活"了 CoT 输出能力

---

## Slide 6 · 解释 #2: 探索增益

随机 reward = 强探索信号
- model 各种尝试
- pretrain 中已有的"好策略"被激活
- 与具体 reward 无关

---

## Slide 7 · 解释 #3: KL ref penalty 的隐藏作用

GRPO 自带 KL ref penalty。
- KL 约束让 actor 不离 SFT 太远
- SFT 已经能做对一些
- → 随机 reward 主要不破坏，正确率 ≈ SFT 上限

---

## Slide 8 · 解释 #4: pretrain 数据已包含答案

Qwen pretrain 见过 MATH dataset (data contamination)。
- model 实际能记答案
- 随机奖励"激活记忆"
- 在 Llama-3-7B 上**不复现** → 因 Llama 没污染

→ contamination 是核心因素。

---

## Slide 9 · 复现矩阵

| base | 真 reward | 随机 reward |
|------|----------|-----------|
| Qwen-2.5-7B | +20 | **+21** |
| Qwen-2.5-1.5B | +15 | +15 |
| Llama-3-7B | +18 | +2 |
| Mistral-7B | +12 | +3 |

→ Qwen 系列 contamination 严重。

---

## Slide 10 · 警示：proxy reward ≠ true reward

经典 RM hacking 的极端版：
- proxy reward (你测的) 涨 → 不代表 gold reward (真本事) 涨
- 随机 reward 也能让 proxy 涨

→ **必须 gold 验证**。

---

## Slide 11 · 怎么避免被骗

```
1. held-out 测试集 (训练绝对不见)
2. 多 base model 对照 (排除 contamination)
3. 多 task 对照 (排除 task-specific 假涨)
4. 人工 spot check (尤其 reasoning quality)
```

---

## Slide 12 · 对 R1 复现的影响

许多 TinyZero / Open-R1 论文报的 +pp 数字现在受质疑：
- 多少是真"学到推理"？
- 多少是激活 CoT？
- 多少是 contamination 助攻？

→ 整个 R1 复现潮被重新评估。

---

## Slide 13 · 工程角度的 takeaway

不要因此放弃 R1 复现，但：
- 报告时分清 "raw improvement" vs "vs base"
- 用多个 held-out + 多 base
- 明确 contamination 风险

---

## Slide 14 · 哲学层面

RL 不是魔法。pretraining 是 90%。RL 只是"激活"。
> "RL teaches the model nothing new; it teaches it which capabilities to exhibit."  
> — Anthropic Scaling Law 2025

---

## Slide 15 · 与 Reward Hacking 关系

L09 (RLHF) 讲过 reward hacking：
- Gao 2022 倒 U 曲线
- 长度偏置 / sycophancy

Spurious Rewards 是终极版：
> "**没有 reward 信号**也能让 proxy 上涨"

→ proxy 与 gold 完全脱钩。

---

## Slide 16 · 一句话总结

> Qwen + 随机奖励 涨 21pp。RL 在 R1 时代的"魔法"50% 来自激活，30% 来自 contamination，20% 来自真学习。**永远 gold 验证**。

下一讲 L13 — Capstone Track A (教学轨)。
