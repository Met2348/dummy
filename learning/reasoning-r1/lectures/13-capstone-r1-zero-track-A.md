# L13 · Capstone Track A — GPT-2-M + Countdown-3 教学轨

> 16 slides | 50 min | R1 时代实操毕业 ⭐

---

## Slide 1 · Track A 目标

**教学**为主：跑通 R1-Zero 完整 pipeline，看 reward 上升 + length 涨。

- base: **GPT-2-medium (355M)** （便宜 + CPU 也能 inference）
- task: **Countdown-3** （比 GSM8K 简单）
- 显存: 单 5090 24GB 充裕
- 时长: 4 算法 × 1.5h = 6h

---

## Slide 2 · 为什么选 GPT-2-M

- 小：单卡放得下 4 model
- 老：pretrain 时无 contamination
- 简单：不会 aha emergence（教学预设说明），只看 pipeline 跑通

→ GPT-2-M 不会出现 R1 那种深度推理，但能学会"用 format + 出对答案"。

---

## Slide 3 · 4 算法横向对照

| algo | 关键 | 预期 |
|------|------|-----|
| 1. REINFORCE + mean baseline | 最朴素 | 不稳定 |
| 2. RLOO (k=8) | 显著稳化 | smooth |
| 3. GRPO (k=8) | + KL | length 开始涨 |
| 4. GRPO + DAPO Clip-Higher | 不对称 clip | aha 早现 |

→ 教学价值：算法演化路径可视化。

---

## Slide 4 · Countdown-3 任务

```python
def gen_problem(rng):
    a, b, c = rng.sample(range(1, 20), 3)
    op1, op2 = rng.choice("+-*"), rng.choice("+-*")
    target = eval(f"{a}{op1}{b}{op2}{c}")
    return [a, b, c], target
```

→ 1000 题预先生成。

---

## Slide 5 · Reward 函数

```
combined = 0.1 * format + 0.9 * accuracy

format:   regex match <think>...</think><answer>...</answer>
accuracy: eval(<answer>) == target ? 1 : 0
```

→ 与 R1-Zero 完全同款。

---

## Slide 6 · 训练超参 (algo 1: REINFORCE)

```yaml
algo: REINFORCE + mean baseline
k: 1 (单 rollout per prompt)
lr: 1e-5
beta_kl: 0
max_response_len: 128
total_steps: 1000
expected: reward 5% → 60% (噪声大)
```

---

## Slide 7 · 训练超参 (algo 2: RLOO)

```yaml
algo: RLOO
k: 8
lr: 1e-5
beta_kl: 0
max_response_len: 128
total_steps: 500
expected: reward 5% → 75% (稳定)
```

baseline = leave-one-out 均值。

---

## Slide 8 · 训练超参 (algo 3: GRPO)

```yaml
algo: GRPO
k: 8
clip_eps: 0.2
beta_kl: 0.04
lr: 5e-6
max_response_len: 256
total_steps: 500
expected: format 95% accuracy 15%, length 50→150
```

---

## Slide 9 · 训练超参 (algo 4: GRPO + Clip-Higher)

```yaml
algo: GRPO + DAPO Clip-Higher
k: 8
clip_eps_low: 0.2
clip_eps_high: 0.28   # 关键
beta_kl: 0.04
lr: 5e-6
max_response_len: 256
total_steps: 500
expected: 同 GRPO 但 length 涨更猛
```

---

## Slide 10 · 监控指标

每 20 step:
- mean_reward
- format_acc
- accuracy
- response_len (mean)
- KL (actor || ref)
- entropy

绘 4 条曲线：4 算法 × 6 指标。

---

## Slide 11 · 预期对照表

| algo | format_acc | task_acc | mean_len |
|------|-----------|----------|----------|
| base GPT-2-M | 5% | 5% | 50 |
| REINFORCE | 60% (噪声) | 10% | 60 |
| RLOO | 80% | 12% | 80 |
| GRPO | 95% | 15% | 150 |
| GRPO+CH | 95% | 17% | 180 |

→ 4 算法路径渐进可见。

---

## Slide 12 · 实战入口

```bash
python learning/reasoning-r1/src/r1_zero_track_a.py \
  --algo grpo --total_steps 500 --k 8
```

train log 写入 `runs/track_a/*.csv`，用 jupyter 画曲线。

---

## Slide 13 · 失败排查

| 现象 | 修 |
|------|---|
| 不学 | lr 调大 → 2e-5 |
| reward 飘 | k 加大 / k=16 |
| OOM | max_response_len 减半 |
| KL 爆 | beta 调大至 0.1 |
| format 学不到 | format 权重 0.1 → 0.3 |

---

## Slide 14 · 与 Track B 关系

Track A 教学（GPT-2-M，pipeline 跑通）；
Track B 真训（Qwen-1.5B，看 aha emergence）.

不强制都跑——Track A 必跑，Track B 选跑。

---

## Slide 15 · Track A 退出条件

- [ ] 4 个算法都跑过 200+ step
- [ ] format reward 收敛到 ≥ 80%
- [ ] task accuracy 显著高于 base (+5pp)
- [ ] length 显著涨 (+50%)
- [ ] 4 条 reward 曲线 plot 完成

---

## Slide 16 · 一句话总结

> Track A 不追 aha，只追"R1 pipeline 跑通 + 4 算法可对照"。教学价值满分。

下一讲 L14 — Track B 挑战轨（Qwen-1.5B + GSM8K）。
