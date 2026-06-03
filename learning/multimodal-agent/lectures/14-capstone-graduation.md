# L14 · ⭐⭐⭐ 毕业 Capstone — 五线综合对照

> 18 slides | 60 min | 整个系列收官

---

## Slide 1 · 系列毕业 ✓

PEFT 三专题 (28 方法) + RL 七专题 (88 方法) = **116 方法 / 90 lecture / ~130h**

```
你已掌握全部:
  ✓ Prompt Tuning Family
  ✓ LoRA Family
  ✓ Adapter Family
  ✓ RL Foundations (PPO)
  ✓ RLHF Classic
  ✓ DPO Family
  ✓ Process Reward
  ✓ R1 时代 (系列高峰)
  ✓ RL SOTA 2026
  ✓ 多模态 + Agent
```

---

## Slide 2 · 毕业作品目标

同一道 GSM8K 题 + **5 个 model** 对照:
1. Vanilla GPT-2 base
2. LoRA fine-tuned (lora-family L01 ckpt)
3. Pfeiffer Adapter (adapter-tuning-family L01 ckpt)
4. DPO aligned (dpo-family L01 ckpt)
5. R1-Zero (reasoning-r1 capstone-A ckpt)

→ 5 种"改大模型"的成果可视化。

---

## Slide 3 · 测试题

```
Q: Janet has 16 eggs. She eats 3 and sells 6 for $2 each.
   How much money does she make from selling eggs?

Ground truth: $12
```

经典 GSM8K 式题，看 5 个 model 的不同推理路径。

---

## Slide 4 · 预期响应 — Vanilla

```
"16 eggs minus some equals 12"
```

无格式，无可靠推理，base 行为。

---

## Slide 5 · 预期响应 — LoRA

```
"Janet eats 3 eggs. She has 16-3=13.
 She sells 6 at $2. Answer: 12"
```

格式 OK，推理浅。LoRA 是 weight 扰动，本质仍是 SFT 后的 base。

---

## Slide 6 · 预期响应 — Adapter

```
"Step 1: 16 - 3 = 13 eggs left.
 Step 2: She sells 6 at $2 = $12.
 Answer: $12"
```

structure 扰动，结构化推理但深度与 LoRA 相当。

---

## Slide 7 · 预期响应 — DPO

```
"To find Janet's earnings:
1. Start: 16 eggs
2. She eats 3, but eating doesn't matter for sales
3. She sells 6 eggs at $2 each
4. Earnings = 6 × $2 = $12
Answer: $12"
```

人类偏好风格强（解释清晰、列点）。distribution shift。

---

## Slide 8 · 预期响应 — R1-Zero

```
<think>
Let me analyze. Janet has 16 eggs. She eats 3 (these are gone, not sold).
She sells 6 of the remaining.
Wait, the question asks about money from selling. So I only need to count sales.
6 × $2 = $12.
Let me verify: 16-3-6=7 left, sold 6, ate 3. Total 16. ✓
</think>
<answer>#### 12</answer>
```

涌现自检 (Wait, verify) + 严格 format。trajectory 改。

---

## Slide 9 · 五线对照矩阵

| Model | 格式 | 准确 | 推理深度 | 自检 | 多解 |
|-------|------|-----|---------|------|-----|
| Vanilla | ✗ | ?  | 无 | ✗ | ✗ |
| LoRA | ✓ | ✓ | 浅 | ✗ | ✗ |
| Adapter | ✓ | ✓ | 浅 | ✗ | ✗ |
| DPO | ✓ | ✓ | 中 | ✗ | ✗ |
| **R1-Zero** | **✓** | **✓** | **深** | **✓** | **✓** |

→ R1-Zero 在所有维度都最强（但显存/训练成本也最高）。

---

## Slide 10 · 与 L13 五线综合对照

L13 已说：
- Prompt: input 扰动
- LoRA: weight 扰动
- Adapter: structure 扰动
- RLHF/DPO: distribution shape 改
- R1: trajectory 改

L14 是 L13 的**实证**：5 个 ckpt 在同题上展现 5 种"改"的效果。

---

## Slide 11 · 工程选型回顾

```
任务类型 → 选 model:
  - 客服 chatbot: DPO ✓
  - 数学竞赛: R1-Zero ⭐
  - 风格化: LoRA
  - 多任务: Adapter
  - 跨语言: Prompt
```

→ 五条线各有最佳场景，无银弹。

---

## Slide 12 · 训练成本对比

| 方法 | 数据 | 时长 | 显存 | 改什么 |
|------|------|-----|------|--------|
| Prompt | 1k | 1h | 8GB | input |
| LoRA | 5k | 3h | 12GB | weight |
| Adapter | 5k | 3h | 12GB | structure |
| DPO | 10k pair | 5h | 14GB | distribution |
| R1-Zero | 5k question | 24h | 22GB | trajectory |

→ R1-Zero 最贵，但产出最强。

---

## Slide 13 · 部署 + 推理成本

| 方法 | 额外参数 | 推理 latency |
|------|--------|------------|
| Prompt | 8k | 1.0× |
| LoRA | 8M | 1.05× |
| Adapter | 50M | 1.1× |
| DPO | 0 (merged) | 1.0× |
| R1-Zero | 0 (merged) | 3× (think 长) |

→ R1-Zero 推理慢 (long CoT)，需配 budget forcing。

---

## Slide 14 · 学完后下一程

2026-06 之后：
1. **MoE 路线**: Mixtral / DeepSeek-MoE / Phi-MoE
2. **长上下文**: 1M / 10M context
3. **World Model RL**: simulator + long-horizon
4. **Agent RL 工业化**: ComputerRL / SWE / Browser
5. **MultiTurn RL**: SCoRe / 多轮自纠错
6. **Multimodal RL**: VLM-R1 后续 (Vision-R1.5/2)

---

## Slide 15 · 学习路径总结

```
2026-01: 启动 (PEFT prompt)
2026-04: PEFT 完结 (28 方法)
2026-06: RL 启动
2026-09: R1 时代峰值
2026-10: 毕业
2026-11+: 自主选 1-2 个方向深耕
```

---

## Slide 16 · 你能做什么

学完此系列后:
- 读所有 2026 RL/对齐论文不卡
- 选 PEFT/RL 算法不靠玄学
- 在 5090 24GB 上跑大部分实验
- 复现 R1-Zero / DPO / DAPO
- 设计自己的对齐方案

---

## Slide 17 · 系列致谢

- DeepSeek (R1)
- ByteDance (DAPO/VAPO)
- HuggingFace (Open-R1/TRL)
- Berkeley (TinyZero)
- Stanford (s1)
- Anthropic (Constitutional AI)
- 你（坚持完整套学习）

---

## Slide 18 · 一句话总结

> 五线综合 = 116 方法 + 5 个 ckpt 对照 = 整个 LLM 适配学的完整图景。

🎓 **系列毕业 ✓**
🎓 **2026-06-04 这一天，你掌握了 2017-2026 LLM 对齐与推理的全部主线方法。**

下一程：自由发挥，深耕一线。
