# L06 · RWKV-7 — Linear Attention 路线

> 20 slides | 60 min ⭐⭐⭐

## Slide 1 · RWKV 简介

```
R: receptance
W: weight decay
K: key
V: value
```

linear attention 变体，无 state matrix (vs SSM)。

## Slide 2 · 数学

```
u_t = receptance(x_t)
w_t = exp(-w_t * t')
k_t, v_t = key(x_t), value(x_t)

out_t = Σ_{t'<t} w_t * k_t' * v_t' * sigmoid(u_t)
```

类似 attention 但用指数衰减替代 softmax。

## Slide 3 · RWKV-7 改进

```
RWKV-6: 已有 linear attention
RWKV-7: 加 time-decay 学习 + gate 多元化
```

## Slide 4 · 与 Mamba 对比

| | Mamba | RWKV-7 |
|---|-------|--------|
| state | yes (d_state) | no (隐含) |
| 实现 | selective scan | weighted sum |
| 推理 | O(d_state) | O(1) |
| ecosystem | mamba-ssm | rwkv lib |

## Slide 5 · 模型规模

```
RWKV-7-0.4B, 1B, 3B, 7B
开源 + 商用 friendly license
```

## Slide 6 · 性能

```
RWKV-7-7B vs Llama-2-7B:
  ppl 接近
  long context 优 (32k+)
  速度更快
```

## Slide 7 · 实现 rwkv_block.py

```python
class RWKVBlock(nn.Module):
    def __init__(self, d):
        self.W_R = Linear(d, d)
        self.W_K = Linear(d, d)
        self.W_V = Linear(d, d)
        self.time_decay = nn.Parameter(...)
        self.time_first = nn.Parameter(...)
    def forward(self, x, state):
        # weighted sum with time decay
        ...
```

## Slide 8 · 多语言友好

RWKV 团队侧重多语言（中文/日文）。tokenizer 优化。

## Slide 9 · 推理超快

O(1) per token (固定隐 state)。
比 transformer KV cache 还省。

## Slide 10 · 训练难

```
parallel form 需 chunked
gradient 在长 seq 上不稳
```

实务训 1B+ RWKV 仍有挑战。

## Slide 11-20 · 详细（略）

## 参考
- RWKV-7 (Peng et al. 2024)
- RWKV GitHub
