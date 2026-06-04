# L06 · 模型架构 (Phi-tiny 270M)

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · Phi-tiny config

```
hidden: 1024
n_head: 16  (GQA: kv_head=4)
head_dim: 64
n_layer: 24
vocab: 50257 (GPT-2 BPE)
seq_len: 2048
intermediate: 4 × hidden = 4096 (SwiGLU)
                ↑ 实际 8/3 × hidden ≈ 2730
```

≈ 270M params.

## Slide 2 · 参数估算

```
embedding:  50257 × 1024 = 51M
attention/layer: 4 × 1024² = 4M  × 24 = 100M
mlp/layer:  3 × 1024 × 2730 = 8M × 24 = 200M
LN, biases: 1M
total ≈ 350M

去掉 embedding (用 tied): ~ 300M
```

## Slide 3 · 标准设计 ablation

| Choice | Phi-tiny |
|--------|---------|
| PE | RoPE base=10000 |
| Norm | RMSNorm |
| Norm position | Pre-norm |
| Activation | SwiGLU |
| KV | GQA (1:4) |
| Tied embedding | Yes |
| LR schedule | WSD |

## Slide 4 · model 代码骨架

```python
class PhiTinyConfig:
    hidden = 1024; n_head = 16; n_kv_head = 4
    n_layer = 24; vocab = 50257; seq_len = 2048

class Block(nn.Module):
    def __init__(self, c):
        self.attn_ln = RMSNorm(c.hidden)
        self.attn = GroupedQueryAttention(c)
        self.mlp_ln = RMSNorm(c.hidden)
        self.mlp = SwiGluMlp(c)
    def forward(self, x):
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x
```

## Slide 5 · SwiGLU MLP

```python
class SwiGluMlp(nn.Module):
    def __init__(self, c):
        d_ff = int(2/3 * 4 * c.hidden)
        d_ff = (d_ff + 63) // 64 * 64
        self.w1 = nn.Linear(c.hidden, d_ff, bias=False)
        self.w2 = nn.Linear(c.hidden, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, c.hidden, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

## Slide 6 · RoPE 注入

```python
def apply_rope(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

## Slide 7 · tied embedding

```python
self.lm_head.weight = self.embedding.weight
```

省 50M 参数 (40% of 270M)。

## Slide 8 · LN scale 残差

```python
nn.init.normal_(self.attn.out_proj.weight, std=0.02 / math.sqrt(2 * n_layer))
nn.init.normal_(self.mlp.w3.weight, std=0.02 / math.sqrt(2 * n_layer))
```

防深网络爆。

## Slide 9 · gradient checkpointing

```python
out = torch.utils.checkpoint.checkpoint(layer, x)
```

每 transformer block 重算 forward，省 activation 1/√L 显存。

## Slide 10 · GQA 节省 KV cache

```
n_head=16, kv_head=4: KV 仅 1/4
推理: 30k ctx 节省 75% KV memory
```

## Slide 11 · 270M 显存估算

```
weights bf16: 540 MB
grad bf16: 540 MB
optimizer state fp32: 2.16 GB (AdamW)
activation: ~ 4 GB (seq 2048 batch 32 + grad ckpt)
total: ~ 7 GB ✓ on 24 GB 5090
```

## Slide 12 · Phi vs Llama 微差

```
Phi tiny:
  - 多用 SwiGLU
  - 数据 textbook 风格
  - 不用 GQA (Phi-1/1.5 是 MHA, Phi-3 才 GQA)
本 capstone 用 GQA (现代)
```

## Slide 13 · 与 transformer-deep L 配合

```
common.py, swiglu.py, rmsnorm.py, gqa.py, rope.py
本 topic 直接复用 (import 或 copy)
```

## Slide 14 · 总结

```
Phi-tiny 270M = Pre-RMSNorm + GQA + RoPE + SwiGLU + tied embed
270M 占 5090 24G 7 GB ✓
```

## 参考
- Phi-3 tech report (Microsoft 2024)
- Llama-3 tech report
