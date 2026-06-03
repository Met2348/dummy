# L06 · Activation — SwiGLU / GeGLU 时代

> 16 slides | 50 min | Transformer Deep 第 6 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 复习 ReLU / GELU / SiLU 性质
2. 理解 GLU 家族（GeGLU / SwiGLU）的"门控"思想
3. 知道为什么 Llama-2 后全用 SwiGLU
4. 写出 30 行 SwiGLU MLP

---

## Slide 1 · 经典激活演化

```
ReLU       max(0, x)              快 / 简单
GELU       x · Φ(x)               GPT-2 后
SiLU       x · sigmoid(x)         Swish 别名
GeGLU      gated GELU              ReGLU 变体
SwiGLU     gated SiLU              事实标准
```

每代提升 0.5-1pp，Llama-2 后 SwiGLU 锁定。

---

## Slide 2 · ReLU

```
ReLU(x) = max(0, x)
```

零成本，但 x<0 完全死。半网络"失活"。

GPT-1 / BERT 早期用。

---

## Slide 3 · GELU

```
GELU(x) = x · Φ(x)   ( Φ 是标准正态 CDF )
       ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715 x³)))
```

平滑 ReLU，x<0 仍有少量"信号"通过。

GPT-2/3 / BERT 用。

---

## Slide 4 · SiLU (Swish)

```
SiLU(x) = x · sigmoid(x) = x / (1 + e^{-x})
```

GELU 与 SiLU 形状极接近，但 SiLU 计算更简单。

Llama-2 用 SiLU 替代 GELU。

---

## Slide 5 · GLU 思想（Dauphin 2017）

```
GLU(x, W, V) = (x W) · σ(x V)
                ↑       ↑
                value   gate
```

输出 = value × gate。gate 像注意力，控制信息流。

---

## Slide 6 · GeGLU

```
GeGLU(x) = GELU(x W_g) · (x W_v)
```

gate 用 GELU 而非 sigmoid。Noam Shazeer 2020 提出。

---

## Slide 7 · SwiGLU

```
SwiGLU(x) = SiLU(x W_g) · (x W_v)
```

gate 用 SiLU。Shazeer 2020 同篇论文。

Llama-2 / Llama-3 / Mistral / Qwen / Phi 全用。

---

## Slide 8 · SwiGLU MLP 全式

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_g = nn.Linear(d, d_ff, bias=False)
        self.w_v = nn.Linear(d, d_ff, bias=False)
        self.w_o = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.w_o(F.silu(self.w_g(x)) * self.w_v(x))
```

注意：3 个 Linear（vs GELU 2 个），但 d_ff 可缩 1.5×。

---

## Slide 9 · d_ff 的选择

```
GELU MLP:    d_ff = 4 · d           (W_in: d → 4d; W_out: 4d → d)
SwiGLU MLP:  d_ff ≈ 8/3 · d ≈ 2.67d  (3 个 W: d → d_ff)
```

参数等价，性能 +0.5pp。

Llama-2 d=4096, d_ff=11008 ≈ 2.69d.

---

## Slide 10 · 性能对比

Llama-1 ablation：

```
GELU MLP   : valid ppl baseline
ReGLU      : -0.2
GeGLU      : -0.3
SwiGLU     : -0.4 ← best
```

差距小但稳定。

---

## Slide 11 · 为什么 GLU 更好

理论上没有清晰证明。猜测：
1. gate 提供数据相关的"动态稀疏性"
2. 增加非线性表达力
3. 模型自己学"哪些维度通过"

实验上稳定，理论仍未完全清楚。

---

## Slide 12 · 实务陷阱

```
W_g, W_v 必须独立，不能 weight tie
bias=False 是标配
d_ff 必须能整除某些 fused kernel 的 multiple of 256
```

---

## Slide 13 · 与 attention 配比

```
LLM transformer block:
  - attention 占 ~ 1/3 参数
  - MLP (SwiGLU)  占 ~ 2/3 参数
```

MLP 是参数大户，激活选择影响显著。

---

## Slide 14 · 推理优化

SwiGLU 的 W_g, W_v 可在推理时 fuse 成一次矩阵乘（QKV 风格）。

```python
# 训练形式
g = self.w_g(x); v = self.w_v(x)

# 推理 fuse
gv = self.w_gv(x)   # (d → 2 d_ff)
g, v = gv.chunk(2, dim=-1)
```

vLLM / TensorRT-LLM 都做。

---

## Slide 15 · 模型 Activation 速查

| 模型 | 激活 |
|------|------|
| GPT-2/3 | GELU |
| Llama-1 | SwiGLU |
| Llama-2/3 | SwiGLU |
| Mistral | SwiGLU |
| Qwen-2.5 | SwiGLU |
| DeepSeek-V3 | SwiGLU |
| Phi-3/4 | SwiGLU |

→ 一致 SwiGLU。

---

## Slide 16 · 课后思考

1. SwiGLU 比 GELU 多 1 个 W，凭什么更好？
2. d_ff = 2.67d 是怎么定的？
3. W_g, W_v fuse 训练时也可以吗？
4. ReGLU (ReLU 版) 为什么不流行？

---

## 参考

- Hendrycks & Gimpel 2016 (GELU)
- Ramachandran et al. 2017 (Swish/SiLU)
- Shazeer 2020 (GLU Variants)
- Llama-2 paper 2023
