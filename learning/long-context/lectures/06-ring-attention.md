# L06 · Ring Attention — Sequence Parallel 1M Context

> 32 slides | 90 min ⭐⭐⭐⭐⭐

> Liu et al. 2023.10 / 多 GPU 序列分片

## Slide 1 · 长 ctx 显存问题

```
attention O(L²) 显存 → 单 GPU 训 128k 难
```

Ring Attention: 把 sequence 切到多 GPU。

## Slide 2 · 基本思路

```
seq L → 切 N 块（每 GPU 1 块）
每 GPU 持有: Q_i, K_i, V_i
Ring 通信: K_j, V_j 在 GPU 间环形传递
```

## Slide 3 · 完整算法

```
for round r in range(N):
    K_curr, V_curr = K[(rank+r) % N], V[(rank+r) % N]
    accumulate score with online softmax
    send K_curr to next rank
```

类似 ring all-reduce。

## Slide 4 · 与 FA 关系

```
FA: 单 GPU 分块 (in SRAM)
Ring: 多 GPU 分块 (跨 GPU)
↓
Ring × FA = 多 GPU FA
```

ring-flash-attention 库实现。

## Slide 5 · 内存

```
单 GPU 全 attn 128k: 80 GB+ → OOM
Ring 8 GPU: 每 GPU 10 GB → OK
```

线性 scale with GPU 数。

## Slide 6 · 通信成本

```
每 round: send K, V (size O(L/N · d))
N round 完成完整 attention
↓
total comm: O(L · d)
```

很小，因为 L 与 d 之积已经是有限值。

## Slide 7 · 实现 ring_attention_naive.py

教学版（单 GPU 模拟多 GPU 通信）。

## Slide 8 · 库 ring-flash-attention

```python
from ring_flash_attention import ring_flash_attn_func
out = ring_flash_attn_func(q, k, v, causal=True)
```

Linux 多 GPU 必备。

## Slide 9 · 1M context 实战

Llama-3 内部用 Ring + FA 训 128k → 1M 部分能力。

## Slide 10 · 与 SSM 对比

```
Mamba (32k 单 GPU): O(L) 自然
Ring + Transformer (1M 多 GPU): O(L²) 但并行
↓
极长 ctx 现仍 Ring + attn 主流
```

## Slide 11-32 · 实现细节 + benchmark（略）

## 参考
- Liu et al. 2023 (Ring Attention)
- ring-flash-attention GitHub
