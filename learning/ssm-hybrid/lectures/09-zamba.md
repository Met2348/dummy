# L09 · Zamba — Shared Attention Hybrid

> 16 slides | 50 min ⭐⭐⭐

## Slide 1 · Zamba 1/2

Zyphra 2024.04 / 2024.10：
- Mamba block 主体
- **共享 attention block** (global, shared across layers)
- 参数高效

## Slide 2 · 共享 attention

```
n_layer × Mamba (各自参数)
+ 1 × Attention (所有层共享)
```

→ 减少 attention 参数 N×。

## Slide 3 · 共享 attention 怎么用

```python
for i, mamba_layer in enumerate(self.layers):
    x = mamba_layer(x)
    if i % 8 == 0:
        x = self.shared_attn(x)   # 同一个 attention!
```

## Slide 4 · 参数效率

```
Zamba-1.2B 总参 1.2B
↓
比 同性能 transformer 少 ~ 30%
```

## Slide 5 · 推理 KV cache

共享 attention 共享 KV cache → 显存优。

## Slide 6 · Codestral-Mamba

Mistral 2024.07 出 Codestral-Mamba：
- 7B 代码模型
- 纯 Mamba（无 attention）
- 性能 ≈ CodeLlama-7B

## Slide 7 · 性能

```
Zamba-1.2B vs Llama-1B:
  ppl 接近
  长 context 优
  推理 快
```

## Slide 8 · 代码

```python
class ZambaModel(nn.Module):
    def __init__(self):
        self.layers = ModuleList(MambaBlock(d) for _ in range(N))
        self.shared_attn = MHA(d)  # 单 attention!
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if (i+1) % K == 0:
                x = self.shared_attn(x)
        return x
```

## Slide 9-16 · 详细（略）

## 参考
- Zamba (Zyphra 2024)
- Codestral-Mamba (Mistral 2024.07)
