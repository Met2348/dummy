# L07 · Sequence Parallel & Context Parallel

> 12 slides | 35 min ⭐⭐⭐

## Slide 1 · 为什么需要 SP

```
TP 切 head, 但 LayerNorm/Dropout/residual 仍全 batch
长 ctx → activation 显存爆
SP: 进一步切 seq 维
```

## Slide 2 · Megatron-SP

```
TP 切 attention/MLP (head 维)
SP 切 LN/Dropout (seq 维)
中间 attention 仍全 seq (借助 all-gather/scatter)
```

显存 1/TP 进一步降到 1/(TP × seq_chunk)。

## Slide 3 · Context Parallel (NVIDIA 2024)

```
attention 也切 seq → CP
配合 Ring Attention / Striped 实现
能训 1M+ ctx
```

## Slide 4 · CP vs SP

```
SP: 只 norm/dropout 切 seq, attention 不切
CP: attention 也切 seq (用 Ring)
更先进
```

## Slide 5 · CP + Ring Attention

```
每 GPU 持有 1/cp 个 seq chunk
attention 经环形 all-to-all 传 K, V
计算完整 attention but 显存仅 1/cp
```

## Slide 6 · CP 实现 (Megatron-Core)

```python
from megatron.core.transformer.context_parallel \
    import context_parallel_forward

out = context_parallel_forward(
    Q, K, V, cp_size=8, cp_rank=rank,
)
```

## Slide 7 · 性能

```
70B + 128k ctx + 8 GPU CP:
  原: OOM
  CP: ✓ 训得动
吞吐 ≈ 80% (通信少)
```

## Slide 8 · DeepSpeed Ulysses

```
长 ctx SP 替代方案
all-to-all 把 seq 维和 head 维换
不依赖 Ring
```

## Slide 9 · 与其它并行组合

```
TP=8 × SP=on × CP=4 × PP=8 × DP=4
= 1024 GPU 训 70B + 1M ctx
```

## Slide 10 · 何时用 SP/CP

```
长 ctx training → CP
短 ctx 但 activation 紧 → SP
小项目 → 不需要
```

## Slide 11 · ckpt vs SP/CP

```
activation checkpointing 可减 90% activation memory
SP/CP 减 1/(TP × CP) 显存
两者乘法叠加
```

## Slide 12 · 总结

```
SP/CP 是长 ctx 训练必备
小项目 (< 8k ctx) 不需要
```

## 参考
- Megatron-SP
- Megatron-CP NVIDIA
- DeepSpeed Ulysses
