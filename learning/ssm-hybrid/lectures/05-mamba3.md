# L05 · Mamba-3 + 长上下文优化

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · Mamba-3

2024 末 / 2025 初推出（社区估计）：
- 长上下文优化
- 多 head SSM
- 与 attention 进一步融合

## Slide 2 · 关键改进

```
1. 多 head SSM (类 GQA)
2. 长序列 chunked SSD
3. 与 attention 混合层
```

## Slide 3 · 多 head SSM

```
原: 1 SSM per channel
Mamba-3: N 个 head 并行 SSM (类似 attention head)
```

→ 表达力 + 并行度提升。

## Slide 4 · 长上下文

```
Mamba 32k context:
  state size: ~ 256-512
  forward time: O(L · d_state²)
```

state size 增大 → 更长记忆。

## Slide 5 · 与 attention 混合

```
某些层 attention，某些层 Mamba
↓
attention 用于全局 retrieval
Mamba 用于流式累积
```

类似 Jamba 设计。

## Slide 6 · 性能

Mamba-3 估计 vs Mamba-2:
- ppl -0.2
- long context (64k+) 强 30%

## Slide 7 · 实现

```python
class Mamba3Block(nn.Module):
    def __init__(self, d_model, n_heads, d_state):
        # n_heads 个并行 SSM
        ...
```

## Slide 8 · 与 R1 推理对接

```
Mamba state 累积 = 类 "rolling context"
→ R1 token rollout 在 Mamba 上可行
```

未来 hybrid R1 训练可能基于 Mamba。

## Slide 9-16 · 详细（略）

## 参考
- Mamba-3 估 (社区/作者后续)
- 长上下文 SSM trends
