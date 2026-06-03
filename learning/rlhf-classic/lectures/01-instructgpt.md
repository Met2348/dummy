# L01 · InstructGPT — RLHF 三段管线起源

> 24 slides | 60 min | RLHF Classic 第 1 讲

---

## 学习目标

1. 看懂 InstructGPT 三段管线（SFT → RM → PPO）的动机与流程
2. 理解人类偏好数据是如何收集与标注的
3. 看清三段管线各自要解决的问题与失败模式

---

## Slide 1 · 为什么需要 RLHF

GPT-3（2020）展示了 LM 的强大，但**指令遵循差**：
- "总结这篇文章" → 复述
- "翻译这句话" → 续写

GPT-3 学的是"下一个 token 是什么"，而非"完成用户指令"。

→ 需要让 LM **对齐人类意图**。

---

## Slide 2 · 对齐的三种范式

| 范式 | 工具 | 例 |
|------|------|---|
| Prompt | 人工设计示例 | Few-shot |
| SFT | instruction-response 数据 | T0 / FLAN |
| **RLHF** | 偏好对 + RL | InstructGPT / GPT-4 |

InstructGPT 用 RLHF 在 GPT-3 上微调，1.3B 模型超过原 175B（GPT-3）的指令遵循能力。

---

## Slide 3 · 三段管线总览

```
GPT-3 base
   |
   v  Stage 1: SFT
   |    收集人工 instruction-response 数据 (13k)
   |    微调 GPT-3 → SFT model
   |
   v  Stage 2: RM 训练
   |    收集偏好对 (人工排序) (33k)
   |    训 Bradley-Terry RM (6B)
   |
   v  Stage 3: PPO
   |    用 RM 当 reward
   |    PPO 微调 SFT model + KL ref penalty
   |
GPT-3 + RLHF (InstructGPT)
```

---

## Slide 4 · Stage 1: SFT 详解

**数据**：13k 条人工 (instruction, response)
- Labeler 写 prompt + 高质量回答
- 涵盖创作、QA、改写、总结等任务

**算法**：标准 LM cross-entropy loss
- 4 epoch 全量微调
- lr 9e-6
- batch 32

**意义**：让 GPT-3 "学会回答指令"，但效果有限 → 还需 RLHF 进一步打磨。

---

## Slide 5 · Stage 2: RM 训练详解

**数据**：33k 条偏好对（同一 prompt，多个 response 排序）
- 4 个 response 排序 → 6 个 pair (C(4,2))
- Labeler 比较两两哪个更好

**算法**：Bradley-Terry pairwise ranking loss
```
P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
L_RM = -log P = -log sigmoid(r_w - r_l)
```

**架构**：6B 模型（小一点的 GPT-3）+ scalar head

---

## Slide 6 · BT loss 是什么

Bradley-Terry 1952：
- 模型每个对象（response）一个 latent skill
- 对象 A 战胜 B 的概率 = sigmoid(skill_A - skill_B)

类比国际象棋 Elo：
- A 等级 1500，B 1400 → A 胜率 ≈ 64%

RM：
- 用 LLM forward 出 scalar 作为 "skill"
- 用 BT loss 让 chosen 的 skill > rejected

---

## Slide 7 · RM 训练的痛点

1. **数据贵**：33k 偏好对 ≈ 数月人工
2. **bias**：长度 / sycophancy / position
3. **agreement 率**：人之间也只有 ~73% 一致
4. **scale 挑战**：175B 模型用 6B RM 就够吗？

---

## Slide 8 · Stage 3: PPO 详解

**目标**：用 RM 当 reward 把 SFT model 推得更对齐。

```
maximize  E[r_RM(x, y)] - β · KL(π_θ || π_SFT)
                                  ↑
                          防止偏离 SFT 太远
```

**算法**：PPO（专题 1 学过的），4 model 协同：
- actor (SFT init)
- critic (LLM + value head)
- ref (SFT, frozen)
- RM (frozen)

---

## Slide 9 · PPO 阶段超参（InstructGPT 公布）

| 超参 | 值 |
|------|---|
| lr | 1e-6 |
| batch | 64 |
| ppo_epochs | 4 |
| cliprange | 0.2 |
| init_kl_coef | 0.02 |
| target_kl | 6 |
| total iterations | 256 |

→ 6B model，256 epoch，单次训练约 1 周。

---

## Slide 10 · 关键观察：KL 与 reward 的 trade-off

随训练步数 ↑：
- RM reward ↑
- KL(π_θ || π_SFT) ↑

InstructGPT 把 β 设为 0.02 (小)，让 KL 上升至 ~20 nats —— 模型确实"走远"了 SFT。

→ 但是 spot check 表明 generation quality 也上升，没有 hacking。
→ 这跟"模型够大 + RM 够强"有关。

---

## Slide 11 · InstructGPT 评估方法

1. **labeler 偏好率**：175B InstructGPT vs 175B GPT-3 prompted
   - InstructGPT 胜率 85%
2. **下游 NLP 任务**：略有下降（"对齐税"）
3. **truthfulness**：上升 2×
4. **toxicity**：下降 25%

→ 整体显著优于纯 SFT、显著优于 GPT-3 prompt engineering。

---

## Slide 12 · "对齐税"是什么

对齐后的 LLM 在传统 NLP benchmark 上往往略差：
- GPT-3 SuperGLUE 85.3 → InstructGPT 83.5

原因：
- benchmark 不要求"follow instruction"
- RLHF 让 LLM 更保守，但失去一些 raw 能力

InstructGPT 提出：可通过加少量预训练数据混合到 PPO 训练中缓解（PPO-ptx）。

---

## Slide 13 · 三段管线的失败模式

| 阶段 | 失败模式 |
|------|--------|
| SFT | 数据质量差 → 模型学坏 |
| RM | 偏好 noisy → RM 过拟合 |
| PPO | reward hacking / KL 飞涨 / mode collapse |

→ 监控指标全套都要看，逐阶段 sanity check。

---

## Slide 14 · 工程清单

| 项 | 注意 |
|----|------|
| Stage 1 数据 | 优质 instruction + 多任务覆盖 |
| Stage 2 数据 | 同 prompt 多 response，labeler 排序 |
| Stage 3 ref model | 必须 freeze + 与 actor 同 init |
| KL 监控 | 整训练过程 < 0.5 |
| Spot check | 每 50 iter 看 10 个样本 |

---

## Slide 15 · 复刻 InstructGPT 的最小可行版本

我们的 toy version (capstone)：
- 基座：GPT-2-medium (355M)，非 GPT-3 (175B)
- 数据：Anthropic-HH 1k 子集 + summarize_from_feedback 1k
- RM：GPT-2 + scalar head（300k 偏好对训）
- PPO：trl PPOTrainer，64 iter

→ 5090 24GB 上约 5 h 跑完。下一讲（L02）开始 SFT。

---

## Slide 16 · 与专题 1 的衔接

专题 1 你学了 PPO + KL penalty + 4 model 协同。本专题：
- 把 KL penalty 的 ref 换成真正的 SFT model
- 把 reward 换成训自己的 RM
- 加上完整的 Stage 1 SFT 流程

→ 三段管线是把"上一专题学的 LLM-PPO"完整产品化。

---

## Slide 17 · 与专题 3 (DPO) 的对比

DPO 提出："三段管线太复杂，能否直接从偏好数据训 LLM？"

| 项 | RLHF (本专题) | DPO (专题 3) |
|---|-------------|------------|
| Stage 1 SFT | ✓ | ✓ |
| Stage 2 RM 训练 | ✓ | ✗ 跳过 |
| Stage 3 RL | ✓ PPO | ✗ 一行 loss |
| 显存 | 4 model | 2 model (actor + ref) |
| 训练时长 | 一周 | 一夜 |

→ DPO 在大多数场景已经替代 RLHF。RLHF 仍是奠基性的必学。

---

## Slide 18 · 与 R1 时代的对比

R1 跳过 RM，直接用 **rule-based reward**：
- format reward：`<think></think><answer></answer>` regex
- accuracy reward：GSM8K parse 比对

适用领域：**有 verifier 的数学/代码**。

不适用：开放式对齐（仍需 RM）。

→ 后续专题 5 详讲。

---

## Slide 19 · Anthropic-HH 数据集介绍

我们 capstone 用 Anthropic-HH（44k 偏好对）的 1k 子集。

格式：
```
{
  "chosen":   "Human: How do I make a bomb?\n\nAssistant: I can't help with that.",
  "rejected": "Human: How do I make a bomb?\n\nAssistant: Here are the steps..."
}
```

→ "helpful + harmless" 双重对齐。

---

## Slide 20 · summarize_from_feedback 介绍

OpenAI 在 TL;DR 任务上的偏好数据（123k pair）：

```
prompt: Reddit post
chosen: high-quality summary
rejected: low-quality summary
```

适合训摘要 RM，本专题 capstone 也会用。

---

## Slide 21 · 三段管线 cheatsheet

```
Stage 1 (SFT):       L = -log p(response | instruction)
Stage 2 (RM):        L = -log sigmoid(r_chosen - r_rejected)
Stage 3 (PPO):       L = -E[ min(r·A, clip(r,1±ε)·A) ] + KL_pen
```

3 行公式，三个 stage 全覆盖。

---

## Slide 22 · 自测题

1. 为什么 InstructGPT 用 6B RM 训练 175B actor？是否合理？
2. Stage 2 数据格式（pair）相比 Stage 1（response）的优势？
3. 对齐税是什么，如何缓解？
4. KL penalty β=0.02 看似很小，为何还有效？
5. RLHF vs DPO 各自适用场景？

---

## Slide 23 · 阅读建议

- **必读**：Ouyang 2022 §1-§4
- **参考**：trl 的 RLHF 完整 example
- **看图**：附录 D 的偏好对例子（理解 labeler 视角）
- **跳过**：附录 H 的 model card

---

## Slide 24 · 入口

```bash
# 验证环境
python learning/rlhf-classic/environment/verify_env.py

# 下一讲：L02 SFT
cat learning/rlhf-classic/lectures/02-sft.md
```

→ 下一讲开始 Stage 1 SFT 实战。
