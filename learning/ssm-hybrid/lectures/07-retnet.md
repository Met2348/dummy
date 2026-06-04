# L07 · RetNet — Retention Mechanism

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · RetNet 定位

```
Microsoft 2023
"Retention" = decay-based attention
```

linear time, parallel training (类 RWKV)。

## Slide 2 · 公式

```
retention(x_n) = (Q_n K_m^T) · D_{n,m}
where D_{n,m} = γ^(n-m)  if n >= m  else 0
```

γ 是 decay 因子（每 head 不同）。

## Slide 3 · 三种 forward

```
parallel form: 类 attention，全 matrix
recurrent:     类 RNN
chunkwise:     混合 (长 seq 用)
```

→ 一个模型多 forward 形式。

## Slide 4 · 与 attention 区别

```
attention: softmax over k
RetNet:    exp decay weighting
```

去掉 softmax → linear, but expressive limited。

## Slide 5 · 与 RWKV-6 关系

```
RWKV-6: decay 由 input 决定
RetNet: decay 固定（learnable per head）
```

RetNet 更简单，RWKV-6 更强表达。

## Slide 6 · 模型

```
RetNet-1.3B, 2.7B, 6.7B (Microsoft)
```

性能与同 size Transformer 相近。

## Slide 7 · 性能 / 速度

```
context 4k:  RetNet ≈ Trans
context 32k: RetNet 3× 更快
推理:        类 RWKV, O(1) per token
```

## Slide 8 · 局限

- 表达力不如 attention
- 工业部署少
- 后被 Mamba / RWKV-7 超越

## Slide 9-16 · 详细（略）

## 参考
- RetNet (Microsoft 2023)
- 与 Mamba/RWKV 对比博客
