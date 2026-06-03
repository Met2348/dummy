# L12 · Capstone — TL;DR 摘要 RLHF

> 12 slides | 30 min | RLHF 完整复刻收官

---

## Slide 1 · Capstone 目标

完整复刻 InstructGPT 三段管线在 **TL;DR 摘要任务** 上。

- 数据：openai/summarize_from_feedback (1k 子集)
- 基座：GPT-2-medium (355M)
- 总时长：5090 24GB ≤ 5h

---

## Slide 2 · 数据准备

```python
from datasets import load_dataset
ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train[:1000]")
```

每条：(post, summary_A, summary_B, preferred)

---

## Slide 3 · Stage 1 SFT (~1h)

- 输入：(post, preferred_summary)
- 1k 条 × 3 epoch
- lr=1e-5，batch=8
- 退出：MT-Bench score > base + 5%

---

## Slide 4 · Stage 2 RM (~1.5h)

- 输入：(post, summary_chosen, summary_rejected)
- BT loss，1k pair × 3 epoch
- 退出：RM accuracy > 60%

```python
loss = -log_sigmoid(RM(c) - RM(r)).mean()
```

---

## Slide 5 · Stage 3 PPO (~2.5h)

- 4 model：actor / critic (with v_head) / ref (SFT) / RM
- rollout 256 responses / step
- 100 step PPO update
- β=0.02, ε=0.2, lr=1e-6

---

## Slide 6 · 关键超参（known-good）

```yaml
sft:   {lr: 1e-5, epochs: 3, batch: 8}
rm:    {lr: 1e-5, epochs: 3, batch: 16}
ppo:
  lr: 1e-6
  beta: 0.02
  clip_eps: 0.2
  rollout_batch: 256
  ppo_epochs: 4
  max_response_len: 64
```

---

## Slide 7 · 监控指标

每 10 step:
- RM-score (proxy reward)
- KL(actor || ref)
- response_len
- entropy

**早停**：KL > 10 立即停（hacking 风险）。

---

## Slide 8 · 预期结果

| 阶段 | 指标 | 预期 |
|------|------|-----|
| SFT | val NLL | 4.5 → 2.8 |
| RM | val acc | 50% → 65% |
| PPO | RM-score | 0.1 → 0.5 |
| PPO | KL | 0 → 5-8 |
| PPO | len | ~30 → ~35 |

---

## Slide 9 · 人工评估

PPO 后 spot-check 10 个：
```
Post: [news article]
SFT output:  [short summary]
PPO output:  [refined summary]
```

应观察：
- ✅ 信息密度提升
- ✅ 关键信息保留
- ❌ 没有过长/重复
- ❌ 没有 hallucinate

---

## Slide 10 · 失败排查

| 症状 | 修 |
|------|---|
| PPO reward 不动 | β 调小至 0.01 |
| KL 飞 | β 加大至 0.05 |
| OOM | response_len 减半 |
| critic loss 不降 | v_head lr × 5 |
| 输出重复 | repetition_penalty=1.2 |

---

## Slide 11 · 进阶：与 DPO 对比

完成后，跑专题 3 的 DPO，**同数据集**：
- DPO：单 model，1.5h 完成
- RLHF：4 model，5h 完成
- 效果：DPO 通常 win 5-10% (offline 数据集)

→ 启示：DPO 在 offline 场景实用，PPO 在 online 探索强。

---

## Slide 12 · 一句话总结

> Capstone 复刻 RLHF 三段，证明掌握 InstructGPT 完整流程。但工程上 5h 训练只是起点，真实工业要 100h+ 反复迭代 RM + actor。

🎓 **Topic 2 RLHF Classic 毕业 ✓**
下一专题 3 — DPO 家族（去 reward model 革命）。
