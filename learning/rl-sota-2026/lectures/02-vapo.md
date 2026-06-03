# L02 · VAPO — Length-Adaptive GAE (ByteDance 2025.04)

> 14 slides | 40 min | DAPO 之后的另一神器

---

## Slide 1 · 动机

DAPO 4 件套之后，ByteDance 继续问：**GAE 的 λ 该不该跟 response 长度走?**

观察:
- 短 response (50 token)：outcome reward 占主导，bootstrap 没意义 → λ 应小
- 长 response (1000 token)：需要 bootstrap 累积信息 → λ 应大

固定 λ 不合理。

---

## Slide 2 · Length-Adaptive GAE 公式

```
λ(y) = clamp(λ_min + α·log|y|, λ_min, λ_max)
```

典型: λ_min=0.5, λ_max=0.99, α=0.1。

| |y| | λ |
|-----|---|
| 10 | 0.73 |
| 100 | 0.96 |
| 1000 | 0.99 |
| 4096 | 0.99 |

---

## Slide 3 · Value Pretraining

VAPO 另一关键：**value head 单独预训 1k step on outcome reward**。

原因：
- GRPO 没 critic，但 VAPO 重新引入 (有 critic 才有 GAE)
- value head 与 actor share backbone 时，随机初始化导致前 1k step 噪声大
- 单独 pretrain 后，PPO loop 稳

---

## Slide 4 · 训练流程

```
Stage 1: Value Pretraining (1k step)
    objective: (V(s) - return)^2
    actor 冻结

Stage 2: VAPO (10k step)
    Length-Adaptive GAE
    + PPO clip
    + Asymmetric KL (用 forward+reverse 平均)
```

---

## Slide 5 · 实测效果

ByteDance Qwen-32B + DAPO baseline + VAPO:
```
DAPO  AIME 50/100
VAPO  AIME 60/100   (+10pp)
```

→ 又一波 10pp。

---

## Slide 6 · 为什么 +10pp

```
1. value pretraining 减少前期噪声: +3pp
2. Length-Adaptive GAE 在长 response 处不丢信号: +5pp
3. Asymmetric KL: +2pp
```

→ 三个独立 trick 累加。

---

## Slide 7 · Asymmetric KL

```
KL_total = (1-γ)·KL(π_actor || π_ref) + γ·KL(π_ref || π_actor)
```

- forward KL 偏 mass-covering
- reverse KL 偏 mode-seeking
- 平均更稳，γ=0.5 推荐

---

## Slide 8 · 与 DAPO 对照

| trick | DAPO | VAPO |
|-------|------|------|
| Clip-Higher | ✓ | (吸收) |
| Dynamic Sampling | ✓ | (吸收) |
| Token-PG | ✓ | (吸收) |
| Overlong Shaping | ✓ | (吸收) |
| Length-Adaptive GAE | ✗ | **✓** |
| Value Pretraining | ✗ | **✓** |
| Asymmetric KL | ✗ | **✓** |

→ VAPO = DAPO + 3 个新 trick。

---

## Slide 9 · 算法位置

```
PPO (2017) → GRPO (2024) → DAPO (2025.03) → VAPO (2025.04)
```

每代加 trick，每代 +5-10pp。但工程复杂度也指数涨。

---

## Slide 10 · 实现要点

```python
def adaptive_lambda(lens, lam_min=0.5, lam_max=0.99, alpha=0.1):
    return (lam_min + alpha * lens.log()).clamp(lam_min, lam_max)

for resp in batch:
    lam = adaptive_lambda(resp.length)
    adv = gae(rewards, values, gamma=1.0, lam=lam)
    ...
```

20 行加在 GRPO 上。

---

## Slide 11 · 何时该用 VAPO

- ✓ 数学推理（长 response，AIME）
- ✓ 代码（复杂 trace）
- ✗ 短对话（< 50 token，length-adaptive 无意义）
- ✗ Countdown 这类玩具（用 DAPO 够）

---

## Slide 12 · 工程坑

| 问题 | 修 |
|------|---|
| value 预训不收敛 | 数据量加大到 10k |
| 长 response GAE 数值不稳 | gamma=0.99 (而非 1.0) |
| KL asymmetric 振荡 | γ=0.5 + 减小 lr |

---

## Slide 13 · 与 verl 的集成

verl 0.4.5+:
```yaml
algorithm:
  adv_estimator: vapo
  adaptive_lambda:
    enabled: true
    lam_min: 0.5
    lam_max: 0.99
  value_pretrain_steps: 1000
```

→ 一行切换。

---

## Slide 14 · 一句话总结

> VAPO = DAPO + Length-Adaptive GAE + Value Pretraining。AIME 50→60。2025 H1 数学推理 RL SOTA。

下一讲 L03 — Dr. GRPO (修 length bias)。
