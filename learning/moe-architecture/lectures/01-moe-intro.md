# L01 · Sparse MoE 起点 — Shazeer 2017

> 18 slides | 55 min | MoE Architecture 第 1 讲 ⭐⭐⭐⭐⭐

---

## 学习目标

1. 理解 MoE 核心思想：稀疏激活
2. Shazeer 2017 LSTM-MoE 起源
3. gating + experts + load balance 三件套

---

## Slide 1 · 为什么 MoE

```
Dense:   每 token → 全 70B 参数都参与
MoE:     每 token → 选 2/256 expert (~3% 激活)
```

→ 参数容量 × 5-10, FLOPs 几乎不变。

---

## Slide 2 · "Sparsely-Gated MoE" (Shazeer 2017)

LSTM 网络替换某层为 MoE：
```
output = Σ_i  G_i(x) · expert_i(x)
G(x) = softmax( top_k(x · W_g) )
```

top-k 选 k 个 expert，其它 G_i=0 → 稀疏。

---

## Slide 3 · Gating 网络

```python
class Gate:
    def __init__(self, d, n_expert):
        self.W = Linear(d, n_expert)
    def forward(self, x):
        logits = self.W(x)
        top_k_logits, top_k_idx = logits.topk(k=2, dim=-1)
        gates = softmax(top_k_logits)
        return gates, top_k_idx
```

---

## Slide 4 · Expert 网络

```python
class Expert:
    def __init__(self, d):
        self.mlp = SwiGLUMLP(d)
    def forward(self, x):
        return self.mlp(x)
```

每 expert 是一个 FFN（与原 dense FFN 同 architecture）。

---

## Slide 5 · 完整 forward

```python
def moe_forward(x, gate, experts):
    gates, idx = gate(x)
    out = zeros_like(x)
    for i, e in enumerate(experts):
        mask = (idx == i).any(dim=-1)
        if mask.any():
            out[mask] += e(x[mask]) * gates[mask, ...]
    return out
```

---

## Slide 6 · "load balance" 问题

如果路由偏向 expert 0：
- expert 0 过载（OOM）
- 其他 expert 闲置（参数浪费）

→ 需要 **aux loss** 强制均衡。

---

## Slide 7 · Shazeer aux loss

```
aux = CV(load)² + CV(importance)²

CV: coefficient of variation (std / mean)
load: 每 expert 接到 token 数
importance: 每 expert 总 gate sum
```

加到总 loss 中：`total = ce + 0.01 * aux`。

---

## Slide 8 · gating noise

为防过早收敛 routing：
```
logits = x · W_g + noise · softplus(x · W_noise)
```

训练时 noise 强，推理时关。

---

## Slide 9 · top-k 选择 k=2

Shazeer 2017 用 k=2：
- k=1: 太稀疏，gradient 流弱
- k=2: gradient 充足，仍稀疏
- k=4+: 太多 expert 激活，省参数效果差

→ 后续 (GShard, Mixtral) 沿用 k=2。

---

## Slide 10 · 参数效率

```
dense 7B FFN:   ~5B param FFN
MoE 7B × 8:     ~40B FFN, 但只激活 ~10B
↓
等效"40B knowledge"模型在 ~10B inference cost
```

---

## Slide 11 · expert capacity

```
capacity = factor × n_tokens × top_k / n_experts
```

factor = 1.25 → 容许 25% overflow。超出的 token "drop"（即不进 MoE）。

---

## Slide 12 · MoE 的"层"

不是每层都改 MoE：
```
Switch Transformer: 每 2 层一 MoE
Mixtral: 每层 MoE
DeepSeekMoE: 前 3 层 dense，后 58 层 MoE
```

dense layers 学通用特征，MoE 学专业知识。

---

## Slide 13 · 训练成本（vs dense）

```
dense 7B vs MoE 8×7B:
  训练 FLOPs:   1× vs 1.2× (略多)
  训练显存:     1× vs 2× (expert state 多)
  收敛速度:     等价 step 数下 MoE 慢
```

但容量等效 × 5，长期 ppl 更好。

---

## Slide 14 · expert 数选择

```
Shazeer 2017:    32, 128, ...
GShard:          ~2k (across multiple GPUs)
Switch:          1k-2k
Mixtral:         8
DeepSeekMoE:     256
```

→ 趋势：少但精（Mixtral）or 多但细（DeepSeek）。

---

## Slide 15 · sparse vs dense FFN params

| 模型 | dense FFN | MoE total | activated |
|------|-----------|-----------|-----------|
| Llama-2 7B | 5B | — | — |
| Mixtral 8x7B | 5B/expert × 8 = 40B | 47B | 13B |
| DeepSeek-V3 | tiny × 256 = 600B | 671B | 37B |

---

## Slide 16 · 教学版 src

`src/moe_layer_naive.py` 含：
- 4 expert 简化版
- top-2 路由
- aux loss
- ~ 100 行注释完整

---

## Slide 17 · 现代 MoE 演化预告

```
L02 GShard (Google 2020)        top-2 + 工程框架
L03 Switch  (Google 2021)        top-1 + 极简
L04 Expert Choice               反向路由
L05 Mixtral (Mistral 2024)      首个开源主流 MoE
L06 DeepSeekMoE (2024)          细粒度 + 共享
L07 Aux-Loss-Free (DS V3 2024)  ⭐⭐⭐⭐⭐ 创新
```

---

## Slide 18 · 课后思考

1. top-k=2 真的最优吗？2024 后期 trends？
2. expert capacity overflow 是否影响 token 学习？
3. MoE 在小模型 (< 1B) 还有意义吗？
4. expert 数 vs expert size 的 trade-off？

---

## 参考

- Shazeer 2017 (Outrageously Large NN with Sparsely-Gated MoE)
- GShard 2020
- Mixtral 2024
