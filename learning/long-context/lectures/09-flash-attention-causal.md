# L09 · FlashAttention 长 ctx 优化

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · FA + 长 ctx

```
FA1/2: 已 O(L) 显存
长 ctx 仍快但有改进空间
```

## Slide 2 · block-sparse FA

```
某些 K 块完全 -inf (causal skip) → 不算
window SWA → 仅算 window 内
```

加速 2-3× 在长 ctx 上。

## Slide 3 · FA + window

```python
flash_attn_func(q, k, v, causal=True,
                 window_size=(4096, 0))
```

只看左 4k context。

## Slide 4 · FA + ALiBi

ALiBi bias 可融进 FA kernel:
```python
flash_attn_func(q, k, v, alibi_slopes=slopes)
```

## Slide 5 · FA3 + 长 ctx

H100 FA3 在长 ctx 上：
- FP8 减半显存
- TMA 加速 K, V 加载
- 32k 上 3× over FA2

## Slide 6-16 · 实务（略）

## 参考
- flash-attn GitHub
- FA3 paper 2024
