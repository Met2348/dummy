# L08 · Infini-Attention

> 22 slides | 65 min ⭐⭐⭐⭐

> Google 2024 / "无限上下文" compressive memory

## Slide 1 · 动机

```
Transformer ctx 限于训练 max_len
Infini: 加 compressive memory state → 理论无限 ctx
```

## Slide 2 · 架构

```
Block(x):
  local_attn = standard MHA on chunk
  memory_state = compressive memory (M, Z)
  retrieval = memory_state · q
  out = local_attn + γ · retrieval
```

local + 全局 memory 双轨。

## Slide 3 · memory state 更新

```
M_new = M_old + sigmoid(K) · sigmoid(V)^T
Z_new = Z_old + sigmoid(K).sum(dim=0)
```

类似 linear attention 累加。

## Slide 4 · retrieval

```
A_mem = sigmoid(Q) · M / (sigmoid(Q) · Z)
```

类似 attention 但用 sigmoid 替 softmax (无归一化)。

## Slide 5 · 优势

```
1. 理论无限 ctx (memory state 固定 size)
2. local attention 仍然准
3. 训练 friendly (不需 ring)
```

## Slide 6 · 局限

```
memory state 有信息损失 (类 SSM)
retrieval 不如 full attention 精确
```

## Slide 7 · 实测

Google 报告：1M ctx 任务 NIAH > 90%。
但 RULER 综合任务表现略弱。

## Slide 8-22 · 详细数学 + 实现（略）

## 参考
- Infini-Attention paper 2024 (Google)
