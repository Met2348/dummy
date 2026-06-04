# L01 · SSM 引论 — HiPPO + State Space

> 24 slides | 70 min ⭐⭐⭐⭐

## Slide 1 · State Space Model

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

控制论经典模型。x: 隐状态，u: 输入，y: 输出。

## Slide 2 · 连续 → 离散

```
x_k = A_bar x_{k-1} + B_bar u_k
y_k = C x_k
```

ZOH (Zero-Order Hold) 离散化：
```
A_bar = exp(Δ A)
B_bar = (A_bar - I)/A · B
```

## Slide 3 · 与 RNN 关系

SSM 离散形式即 linear RNN：
```
h_t = A h_{t-1} + B x_t
```

但 A 用 HiPPO 矩阵 → 长记忆。

## Slide 4 · HiPPO 矩阵

Gu 2020：使 x 是输入历史的多项式投影。

```
A_{nk} = -(2n+1)^{1/2} · (2k+1)^{1/2}     if n > k
A_{nn} = -(n+1)
```

长记忆能力（不爆炸 / 不消失）。

## Slide 5 · S4 (2022)

```
x = HiPPO state
input u → output y via SSM
卷积形式: y = K * u
where K = A_bar^t × B_bar
```

S4 用 FFT 算长卷积 → O(L log L)。

## Slide 6 · 计算两种形式

```
recurrent:  O(L · d_state)        训练慢但推理快
convolution: O(L log L · d_state)  训练快 (FFT)
```

S4 训练用卷积，推理用 recurrent。

## Slide 7 · S5 (2022)

S4 用多种 trick (diagonal A)，S5 简化：
```
A = diag(λ_i)     all complex eigenvalues
recurrent + parallel scan
```

更易实现，性能略低于 S4 但工程友好。

## Slide 8 · selective scan (Mamba)

S4 / S5 是 LTI（time-invariant）。
Mamba: A, B, C 随 input 变化 → **selective**。

```
A_t = A_base + f(u_t)
B_t = g(u_t)
```

→ 失去卷积形式（不再 LTI），需新算法。

## Slide 9 · 长上下文性质

```
attention: O(L²) 显存
SSM:       O(L) 显存 (固定 state size)
```

→ SSM 天然长上下文友好。

## Slide 10 · 性能对照

```
Mamba 1.4B vs Transformer 1.4B:
  ppl 接近
  long context 优 (8k+)
  速度更快 (32k 上)
```

## Slide 11 · 实现 naive recurrent

```python
def ssm_recurrent(u, A, B, C, delta):
    A_bar, B_bar = discretize(A, B, delta)
    x = zeros(d_state)
    y = []
    for t in range(L):
        x = A_bar * x + B_bar * u[t]
        y.append((C * x).sum())
    return stack(y)
```

## Slide 12 · 与 attention 互补

```
attention: 全局精确，但 O(L²)
SSM:       linear time, 但 state size 限信息容量
```

→ Hybrid (Jamba) 组合两者。

## Slide 13 · 与 RWKV 比较

```
RWKV: linear attention 变体，无 state
SSM:  显式 state
```

两条路线收敛于 linear-time sequence modeling。

## Slide 14 · 学到什么

SSM 是 attention 之外的 sequence 范式。
对长序列特别有效。
未来主流可能 hybrid。

## Slide 15-24 · 详细推导（略 — 见 S4 论文）

主要公式：
```
HiPPO: A_{nk}
ZOH: A_bar, B_bar
parallel scan: associative
```

## 参考
- HiPPO (Gu et al. 2020)
- S4 (Gu et al. 2022)
- S5 (Smith et al. 2022)
