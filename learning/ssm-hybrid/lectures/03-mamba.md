# L03 · Mamba — Selective SSM

> 32 slides | 90 min ⭐⭐⭐⭐⭐ 必修

> Gu & Dao 2023.12 / "Transformer 替代品" 之一

## Slide 1 · Mamba 概览

```
S4 是 LTI (Linear Time-Invariant)
Mamba (S6) 让 A, B, C 随输入变 → selective
```

→ 失去卷积形式，但可"忘记 / 记住"。

## Slide 2 · 核心 motivation

attention 优：可"按需 attend"。
S4 局：所有 token 同样处理。
→ Mamba 让 SSM 也有"按需"能力。

## Slide 3 · Selection mechanism

```
原 S4: A, B, C 固定
Mamba:
  delta_t = f_delta(u_t)
  B_t = f_B(u_t)
  C_t = f_C(u_t)
```

每 token 计算自己的 delta, B, C。

## Slide 4 · Mamba block

```
Block(x):
  x = norm(x)
  x = conv1d(x)
  delta, B, C = projections(x)
  y = selective_scan(x, delta, A, B, C)
  return out_proj(silu(y))
```

## Slide 5 · selective_scan kernel

```
selective_scan 是 Mamba 的 CUDA 核心
parallel scan 算 y
A 不变, B/C/delta 变
```

mamba-ssm 库提供 cuda kernel。

## Slide 6 · A 矩阵特殊

```
A = -exp(A_log)   # diagonal, all negative
```

负实数 → 稳定（不爆炸）。

## Slide 7 · 推理 vs 训练

```
训练: parallel scan (O(L log L))
推理: recurrent (O(d_state) per token)
```

KV cache 替代为 SSM state（小很多）。

## Slide 8 · 显存

```
Mamba 1.4B vs Transformer 1.4B:
  context 8k:
    Mamba state: ~ 1 MB
    Trans KV: ~ 500 MB
```

## Slide 9 · 速度

```
context 4k: Mamba 与 trans 同
context 32k: Mamba 3-5× 更快
context 128k: Mamba 仍 O(L), trans OOM
```

## Slide 10 · 性能

```
Mamba 130M vs Pythia 130M:
  ppl 同
  long context (Project Gutenberg): Mamba +10%
  HellaSwag, ARC: 相近
```

## Slide 11 · 模型

```
Mamba-130M, 370M, 790M, 1.4B, 2.8B
```

Gu 团队开源，可加载使用。

## Slide 12 · Mamba 局限

```
1. 不能"回看"完整 context (state 有损)
2. multi-modal 支持弱
3. fine-tune ecosystem 不如 Transformer 成熟
```

## Slide 13 · 代码 mamba_block.py

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        self.in_proj = Linear(d, 2*d)   # gate path + main
        self.conv = Conv1d(d, d, d_conv)
        self.x_proj = Linear(d, dt_rank + 2*d_state)
        self.dt_proj = Linear(dt_rank, d)
        self.A_log = nn.Parameter(log(arange(1, d_state+1)).repeat(d))
        self.D = nn.Parameter(ones(d))
        self.out_proj = Linear(d, d)
```

## Slide 14 · forward 详细

```python
def forward(self, x):
    b, t, d = x.shape
    xz = self.in_proj(x)
    x, z = xz.chunk(2, dim=-1)
    x = silu(self.conv(x.transpose(1,2))[:, :, :t].transpose(1,2))
    x_dbl = self.x_proj(x)
    dt, B, C = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
    dt = softplus(self.dt_proj(dt))
    A = -exp(self.A_log)
    y = selective_scan(x, dt, A, B, C)
    y = y * silu(z)
    return self.out_proj(y)
```

## Slide 15 · naive selective_scan (Python)

```python
def selective_scan_naive(u, dt, A, B, C):
    b, t, d = u.shape
    d_state = A.shape[-1]
    h = zeros(b, d, d_state)
    out = []
    for i in range(t):
        dA = exp(dt[:, i, :, None] * A)
        dB = dt[:, i, :, None] * B[:, i, None, :]
        h = h * dA + dB * u[:, i, :, None]
        y = (h * C[:, i, None, :]).sum(-1)
        out.append(y)
    return stack(out, dim=1)
```

慢但教学清晰。

## Slide 16 · mamba-ssm 库

```python
from mamba_ssm import Mamba
m = Mamba(d_model=512, d_state=16, d_conv=4)
out = m(x)  # uses CUDA kernel
```

CUDA kernel ~ 100× 快于 naive。

## Slide 17 · 与 attention 数学等价吗

不等价。
- attention: 全 pairwise
- Mamba: state 累积 (有损)

但实务 ppl 相近，因为 state size 64 已经够。

## Slide 18 · "state 信息容量"

```
state size 16: 短期记忆 16 维
state size 64: 中期
state size 256: 长期 (Mamba-2 用)
```

太大 → 训练慢；太小 → 长记忆差。

## Slide 19 · Mamba-1 vs Mamba-2

```
M1: selective scan kernel
M2: SSD (State-Space-Duality), 矩阵形式 + 与 attention 等价
```

M2 训练快 2-3×。

## Slide 20 · Mamba 论文成绩

```
Pile train, evaluate on:
  HellaSwag: 38 (Mamba-1.4B) vs 36 (Pythia)
  PIQA:      74 vs 70
  WinoGrande: 56 vs 53
```

## Slide 21 · 实务何时用

```
长 context (32k+):    Mamba 友好
高并发推理:           Mamba 友好 (state 小)
通用任务:             Transformer 仍主流
```

## Slide 22 · 加载 Mamba

```python
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
```

仅 WSL2 + Linux + 对应 GPU 才装得上。

## Slide 23 · 多模态 Mamba

VMamba (vision)、AudioMamba 等出现。
但效果不如 Mamba-language。

## Slide 24-32 · 详细实现、benchmark、与 Jamba 对比（略）

## 参考
- Mamba (Gu & Dao 2023.12)
- mamba-ssm GitHub
