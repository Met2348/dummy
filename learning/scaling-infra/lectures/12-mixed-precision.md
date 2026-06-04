# L12 · Mixed Precision + Stability

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 为什么混精度

```
fp32 训练: 显存 2×, 算力 1×
bf16/fp16 训练: 显存 1×, 算力 2-4×
但精度 trade-off
```

## Slide 2 · fp16 vs bf16

```
fp16: 1 sign + 5 exp + 10 mantissa
  动态范围 6e-8 ~ 65504  (窄)
bf16: 1 sign + 8 exp + 7 mantissa
  动态范围 same as fp32  (大)
```

LLM 训练 → **必 bf16**。

## Slide 3 · loss scaling (fp16 only)

```
fp16 forward 小 grad 下溢 → 0
loss × 2^16 → bwd → 非零 grad → 除回
PyTorch GradScaler 自动
bf16 不需要
```

## Slide 4 · PyTorch AMP

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    out = model(x)
    loss = criterion(out, y)
scaler.scale(loss).backward()  # bf16 不需 scale
optimizer.step()
```

## Slide 5 · 哪些层保留 fp32

```
LayerNorm: bf16 OK
softmax:  bf16 一般 OK, 数值敏感时 fp32
loss:     必 fp32
optimizer state: 必 fp32
```

## Slide 6 · loss spike (训崩)

```
现象: loss 突然 ↑↑↑ 永不回 (或 nan)
原因: bf16/fp16 数值不稳 + grad outlier
位置: attention softmax 出现 inf
```

## Slide 7 · 缓解 loss spike

```
- gradient clipping: clip_grad_norm 1.0
- learning rate warmup 长一点
- mixed precision: norm 留 fp32
- skip step if loss > N × ema_loss
```

## Slide 8 · Z-loss / router-z-loss (MoE)

```
attention/router output logits → softmax 前
加 lambda * (logsumexp - 0)^2
防止 logit 极端化
```

## Slide 9 · fp32 master weights

```
ZeRO/FSDP 默认: weight bf16 + master fp32 copy
grad accumulate 在 fp32
更新后 cast 回 bf16
```

## Slide 10 · Activation 精度

```
gradient checkpointing 保存 bf16 activation
重算时 bf16
若不稳: 用 fp16 weight + bf16 activation
```

## Slide 11 · stable AdamW

```python
torch.optim.AdamW(eps=1e-8)
# 大 batch 用 eps=1e-5 防 division
```

## Slide 12 · 调试

```
log: loss, grad_norm, param_norm
spot check: 训前 50 step grad_norm 不应爆
if grad_norm > 100: 检查 lr / batch / data
```

## 参考
- PyTorch AMP
- bfloat16 (Wang 2019)
- Loss spike studies (PaLM tech report)
