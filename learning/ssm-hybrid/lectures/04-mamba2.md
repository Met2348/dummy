# L04 · Mamba-2 — SSD (State Space Duality)

> 24 slides | 70 min ⭐⭐⭐⭐

> Mamba-1 → Mamba-2 (2024.06)，理论统一 + 工程提速

## Slide 1 · 动机

Mamba-1 缺点：
- selective_scan kernel 复杂
- 与 attention 关系不清

→ Mamba-2 重新表述为矩阵形式（SSD）。

## Slide 2 · SSD 定义

```
y = SSM(A, B, C) · u
matrix form: Y = (L ⊗ A) · M(B, C) · U
```

`L`: 下三角 mask, `M`: 由 B, C 构成的矩阵。

## Slide 3 · 与 attention 等价性

```
attention: Y = softmax(QK^T) V
SSD:       Y = M(B, C) · U  (M 类比 QK^T)
```

数学上 Mamba-2 与"masked linear attention"等价。

## Slide 4 · 训练快 2-3×

```
M1 kernel: selective_scan (custom CUDA)
M2:        矩阵乘 (vendor BLAS 已优化)
```

→ 标准 matmul kernel 利用更充分。

## Slide 5 · 实现差异

```python
# Mamba-1
y = selective_scan(u, dt, A, B, C)

# Mamba-2
chunked SSD:
  for chunk in chunks:
    Y_chunk = chunk_matmul(...)
```

chunk size ~ 64-256，平衡显存与并行。

## Slide 6 · 与 attention "duality"

```
attention 在每 query 看全 keys
SSD 在每 token 累积 + 看历史 (state)
```

数学等价的两种视角。

## Slide 7 · 模型

```
Mamba2-130M, 370M, 780M, 1.3B, 2.7B
```

性能比 Mamba-1 同 size +0.5pp。

## Slide 8 · 与 attention 混合更容易

Mamba-2 矩阵形式与 attention 接口接近，便于 hybrid。

## Slide 9 · 代码 mamba2_block.py

```python
class Mamba2Block(nn.Module):
    def forward(self, x):
        # 类 Mamba-1 但用 SSD chunk
        ...
```

实务用 mamba-ssm lib `Mamba2` 类。

## Slide 10 · 推理

Mamba-2 推理与 Mamba-1 一样 (recurrent O(d_state))。

## Slide 11 · benchmark

Mamba-2-2.7B vs Mamba-1-2.7B:
- ppl -0.3
- 训练时间 -40%
- 推理 同

## Slide 12 · 局限

- 与 attention 仍非完全等价
- 长序列 chunk size 选择难

## Slide 13-24 · 详细推导和代码（略 - 见论文 Algorithm 1）

## 参考
- Mamba-2 (Dao & Gu 2024.06, ICML)
