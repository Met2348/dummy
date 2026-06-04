# L04 · 初始化与 LR Schedule

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · 初始化

```
embedding: N(0, 0.02)        # GPT-2 默认
linear:    N(0, 0.02 / √(2L))  # 深网络残差缩放
LN gamma:  1
LN beta:   0
```

L = 层数。

## Slide 2 · μ-Parametrization (μP)

```
Yang 2022 (Tensor Programs V)
小模型搜超参 → 大模型直接复用
关键: weight_lr 按 width scale
```

## Slide 3 · μP 公式

```
fan_in scale:
  linear  init: σ ∝ 1/√fan_in
  linear   lr: lr ∝ 1/fan_in
  output proj: σ ∝ 1/fan_in  (额外缩放)
embedding:
  init σ: 常数 (不随 width)
```

## Slide 4 · learning rate 大小

| 模型 | base lr | warmup |
|------|---------|--------|
| GPT-2 124M | 6e-4 | 2000 |
| GPT-3 175B | 6e-5 | 375M token |
| Llama-2 7B | 3e-4 | 2000 |
| Llama-3 70B | 1.5e-4 | 8000 |
| Phi-2 | 2.5e-4 | 1000 |

经验: lr ∝ 1/√N (μP) 或 1/N (Adam scale).

## Slide 5 · warmup

```python
def warmup(step, warmup_step, base_lr):
    return base_lr * step / warmup_step
```

省略 warmup → loss spike 风险大。

## Slide 6 · cosine schedule

```
lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))
```

Llama / Qwen 标配。

## Slide 7 · WSD (Warmup-Stable-Decay)

```
warmup: 5%
stable: 75% (lr = base_lr)
decay:  20% (linear to 0)
```

Phi/MiniCPM 用。优势: stable 阶段可 resume + 继续训。

## Slide 8 · WSD vs cosine 对比

```
end-of-train loss:
  cosine: 2.5
  WSD:    2.4 (annealing 集中)
```

但 WSD 中段 ckpt 比 cosine 高 (更高 lr)。

## Slide 9 · 1/√t (Inverse Sqrt)

```
lr(t) = max_lr * √(warmup / t)  for t > warmup
```

T5 / 早期 transformer 用。现代少见。

## Slide 10 · LR vs Batch (Linear scaling)

```
batch ↑ k 倍 → lr ↑ k 倍 (有上限)
LayerSwitch: 上限 lr ≈ 5e-4 for AdamW
```

## Slide 11 · weight decay

```
AdamW: weight_decay 0.1
LN / bias: 不要 decay
embedding: 不要 decay (会损失记忆)
```

## Slide 12 · LR sweep 经验

```
小 ckpt: lr 1e-3
模型变大: lr 减
batch 变大: lr 增
spike → 减半
```

## Slide 13 · gradient clipping

```
clip = 1.0 (Llama 默认)
loss spike 多 → 减到 0.5
```

## Slide 14 · 总结

```
init: GPT-2 style 0.02 + LN 残差缩放 / μP
schedule: cosine 是 baseline, WSD 更佳
warmup + clip 必备
```

## 参考
- Yang 2022 μP
- WSD (MiniCPM 2024)
