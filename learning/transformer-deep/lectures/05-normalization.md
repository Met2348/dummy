# L05 · Normalization — LayerNorm / RMSNorm / Pre vs Post

> 18 slides | 55 min | Transformer Deep 第 5 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 推 LayerNorm vs RMSNorm 公式
2. 理解 Pre-LN vs Post-LN 稳定性
3. 知道 DeepNorm 解决了什么

---

## Slide 1 · LayerNorm (Ba 2016)

```
μ = mean(x)
σ = std(x)
LN(x) = γ · (x - μ) / σ + β
```

per-token，跨 d_model 维度。参数 γ, β (d_model,)。

---

## Slide 2 · RMSNorm (Zhang & Sennrich 2019)

```
RMS(x) = sqrt( mean(x²) )
RMSNorm(x) = γ · x / RMS(x)
```

省掉均值减法 + bias β。等价于"假设 x 是 0-mean"的简化。

---

## Slide 3 · 为什么 RMSNorm

实验：在 LLM 中 mean(x) 接近 0 (init 后)。所以减 μ 几乎没用。

RMSNorm:
- 速度 +7-10%
- 参数 -50% (γ only)
- 性能 ≈ LayerNorm

→ Llama / GPT-NeoX / Phi / Qwen 全用 RMSNorm。

---

## Slide 4 · Pre-LN 公式

```
x = x + Sublayer(LN(x))
```

LN 在 sublayer **之前**。

---

## Slide 5 · Post-LN 公式

```
x = LN(x + Sublayer(x))
```

LN 在 residual 之后（Vaswani 2017 原版）。

---

## Slide 6 · Pre vs Post 稳定性

```
Post-LN:  N 层 lr 衰减需 ~ N^{-1/2}（脆弱）
Pre-LN:   N 层即使 lr 大也稳（不脆弱）
```

→ Pre-LN 已成事实标准。

---

## Slide 7 · 数学直觉

Post-LN 每层都"重置" residual scale；Pre-LN 让 residual 路径全程畅通。

Llama-2: 总是 `h = h + attn(RMS(h))`。

---

## Slide 8 · DeepNorm (Wang 2022)

针对 Post-LN 稳定性问题：

```
x = LN( α · x + Sublayer(x) )
```

α > 1 的 scale factor，让 residual 在深网络下不被淹。

适用 1000 层 transformer 训练。

---

## Slide 9 · 三种 placement 对照

| | Pre-LN | Post-LN | DeepNorm |
|---|--------|---------|----------|
| 稳定性 | 高 | 低 | 极高 |
| 性能 | 标 | 标 | 标 |
| 深度 | ≤ 100 | ≤ 24 | 1000+ |
| 主流 | ✓ | × | 罕见 |

---

## Slide 10 · 实现 LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mu) / (std + self.eps) + self.beta
```

---

## Slide 11 · 实现 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.gamma * x * rms
```

---

## Slide 12 · 数值精度

```
LN / RMSNorm 内部应在 fp32 算（即使输入 bf16）
otherwise: σ²=mean(x²) 在 bf16 overflow / underflow
```

PyTorch nn.LayerNorm 默认在 fp32 计算（autocast off）。

---

## Slide 13 · epsilon 选择

```
LayerNorm: 1e-5（PyTorch 默认）
RMSNorm:   1e-6 (Llama)
```

小 → 早期训练不稳；大 → 后期收敛慢。Llama-2 实测 1e-6 最稳。

---

## Slide 14 · 与 weight init 配合

```
Pre-LN 的 init 一般 N(0, 0.02)
RMSNorm.gamma init = 1
Linear.weight = N(0, 0.02 / sqrt(2N))   # depth-scaled
```

GPT-2 / Llama 都按此 init。

---

## Slide 15 · 现代 LLM Norm 配方

| 模型 | Norm |
|------|------|
| GPT-2 | LayerNorm Post |
| GPT-3 | LayerNorm Pre |
| Llama-1/2/3 | RMSNorm Pre |
| Mistral / Qwen | RMSNorm Pre |
| DeepSeek-V3 | RMSNorm Pre |

→ 一致：RMSNorm Pre-LN。

---

## Slide 16 · PyTorch 2.5 nn.RMSNorm

```python
import torch.nn as nn
norm = nn.RMSNorm(d_model, eps=1e-6)
```

官方实现，CUDA kernel 加速。Pytorch 2.5+ 可用。

---

## Slide 17 · 跨头 / 跨 batch norm

实务 LLM 不用 BatchNorm（不适合自回归）。
LayerNorm / RMSNorm 是 per-token operation，与 batch / seq 无关。

GroupNorm 用于 vision，不进 LLM。

---

## Slide 18 · 课后思考

1. RMSNorm 假设 mean(x)=0，违反时会怎样？
2. Pre-LN 在 100+ 层为什么仍稳？
3. eps 太大对推理有影响吗？
4. PyTorch nn.LayerNorm 内部 fp32 的成本？

---

## 参考

- Ba 2016 (LayerNorm)
- Zhang & Sennrich 2019 (RMSNorm)
- Wang 2022 (DeepNorm)
- Llama-2 paper 2023
