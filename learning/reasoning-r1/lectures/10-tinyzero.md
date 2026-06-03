# L10 · TinyZero — $30 复现 aha moment

> 14 slides | 35 min | R1 复现潮代表作

---

## Slide 1 · 故事

2025.01.30，DeepSeek R1 后 10 天，UC Berkeley 的 Jiayi Pan 发布 TinyZero:
- $30 训练成本
- Qwen2.5-3B
- 复现 aha moment

转推过万。

---

## Slide 2 · 设置

```
base:   Qwen2.5-3B
task:   Countdown / simple multiplication
reward: format + accuracy
algo:   PPO (后续切 GRPO)
infra:  2 × H100 PCIe (rent)
time:   ~6h
cost:   $30 ($2.5/h × 2 GPU × 6h)
```

---

## Slide 3 · 为什么选 Countdown

不选 GSM8K 的原因：
- Countdown 完全规则（无歧义）
- 短题（30 token vs GSM8K 100+）
- 多解性（鼓励探索）
- 评估快

→ 适合 demo aha emergence。

---

## Slide 4 · Countdown 任务

```
Question: Use the numbers [a, b, c] and +, -, *, / to make target T.
Answer:   ((a + b) * c) = T
```

verifier 简单：eval 表达式 == T？

---

## Slide 5 · 训练曲线 (复现)

```
                response_len  format_acc  task_acc
step 0:         50            5%          5%
step 500:       80            70%         30%
step 1000:      140           90%         55%
step 1500:      200           95%         75%
step 2000:      220           96%         80%
```

→ task accuracy 从 5% → 80%。

---

## Slide 6 · aha moment 涌现 (step ~1000)

模型开始输出：
- "Let me try a different approach"
- "Wait, that's not right..."
- "Let me reconsider..."

→ pretrain 中存在的 CoT 模板被激活。

---

## Slide 7 · 复现的意义

**关键洞见**：
- 不需要 671B base
- 不需要 1024 H800
- **个人 dev 用 $30 也能跑通**

→ 把 R1 范式"民主化"。

---

## Slide 8 · TinyZero 局限

```
✓ format + accuracy 显著提升
✓ aha moment 涌现 (词频)
✗ 在 GSM8K 上有限提升 (Countdown 太简单)
✗ 长 CoT 不及大模型
✗ 复杂数学崩
```

→ 教学价值 > 工业价值。

---

## Slide 9 · 后续：Mini-R1 / SimpleRL

```
Mini-R1 (Phil Schmid):
    Qwen-1.5B + GSM8K + GRPO
    教学完美，5090 24GB 可跑

SimpleRL-Zoo (HKUST):
    多 base 对照 (Llama/Qwen/Mistral)
    Llama 失败 → contamination 证据
```

---

## Slide 10 · 复现关键参数（合集）

| 参数 | TinyZero | Mini-R1 |
|------|----------|---------|
| base | Qwen-3B | Qwen-1.5B |
| algo | PPO→GRPO | GRPO |
| k | 8 | 4 |
| max_resp_len | 1024 | 512 |
| lr | 5e-6 | 1e-5 |
| temperature | 0.7 | 0.8 |
| epochs | 4000 step | 1000 step |

---

## Slide 11 · 显存技巧

5090 24GB 跑 Qwen-1.5B GRPO:
```
4bit quant (bitsandbytes nf4)
+ LoRA r=16 on q/k/v/o
+ max_response_len = 256
+ rollout_batch = 8
+ ppo_epochs = 2
+ gradient checkpointing
```

→ 显存 ~18GB.

---

## Slide 12 · 复现成本对比

| 模型 | 算力 | 成本 |
|------|------|-----|
| DeepSeek-R1 | 1024 H800 × 7 day | $5M |
| Open-R1 (HF) | 64 H100 × 3 day | $50k |
| TinyZero | 2 H100 × 6h | $30 |
| **Mini-R1** | **1 RTX 4090 × 24h** | **$0 (own)** |

→ 个人可跑。

---

## Slide 13 · 评估 aha emergence

aha 词频统计：
```python
AHA = ["wait", "let me reconsider", "actually", "rethink", ...]
ratio = responses_with_aha / total
```

合格阈值：≥ 5%。

---

## Slide 14 · 一句话总结

> TinyZero 证明 R1 范式不是大厂专利。$30 + Qwen-3B + Countdown = aha emergence。

下一讲 L11 — Open-R1 HuggingFace 完整开源路线。
