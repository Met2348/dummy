# Lecture 4: Parallel Adapter

> 论文: He et al. 2022, "Towards a Unified View of Parameter-Efficient Transfer Learning" (ICLR)
> 配套代码: `src/parallel_minimal.py` + `src/parallel_adapters.py`
> 配套 notebook: `notebooks/04-parallel-adapter.ipynb`

---

## Slide 1: 本节路线

```
痛点 (串联慢) → 并联 idea → 结构对比 → LoRA 视角 → 代码 → 思考
```

**学完本节，你应该能**:
1. 解释串联 vs 并联结构差异
2. 推导 Parallel Adapter 与 LoRA 的等价关系（无非线性时）
3. 直观理解为什么并联是"Towards a Unified View"的前奏

---

## Slide 2: 痛点 — 串联结构的问题

**之前所有 adapter 都是串联**:
$$h \xrightarrow{\text{base}} h' \xrightarrow{\text{adapter}} h''$$

**问题**:
1. **不可并行计算**: adapter 必须等 base 算完
2. **梯度信号弱**: adapter 的梯度要穿过 base 才能到达 input
3. **难合并**: 串联的非线性阻碍权重合并

**He et al. 的观察**: 也许并联更好。

---

## Slide 3: 并联结构

$$
h \xrightarrow{\text{base}} \text{base}(h) \quad\quad h \xrightarrow{\text{adapter}} \text{adapter}(h)
$$
$$
\text{out} = \text{base}(h) + s \cdot \text{adapter}(h)
$$

```
            ┌──→ base(h) ──┐
        h ──┤              ├──→ out
            └──→ adapter(h)─┘ × scaling
```

**特点**:
- adapter 与 base 并行计算（GPU 利用率高）
- 梯度直接到 input（不经过 base）
- scaling 因子 $s$（类似 LoRA 的 $\alpha/r$）

---

## Slide 4: Parallel Adapter 公式

$$\text{out} = \text{base}(x) + s \cdot W_{up} \sigma(W_{down} x)$$

其中:
- $W_{down} \in \mathbb{R}^{r \times d}$: 降维
- $W_{up} \in \mathbb{R}^{d \times r}$: 升维（零初始化）
- $\sigma$: GELU
- $s$: scaling（默认 1.0，论文用 4.0）

**对比串联**:
$$\text{out} = \text{adapter}(\text{base}(x)) = \text{base}(x) + W_{up} \sigma(W_{down} \cdot \text{base}(x))$$

→ **关键差异**: $\sigma$ 内是 $x$ vs $\text{base}(x)$。

---

## Slide 5: 与 LoRA 的关系

**Parallel Adapter** (有非线性):
$$\Delta = W_{up} \cdot \sigma(W_{down} \cdot x)$$

**去掉 $\sigma$**:
$$\Delta = W_{up} \cdot W_{down} \cdot x = (W_{up} W_{down}) x = \Delta W \cdot x$$

这就是 **LoRA**！其中 $\Delta W = W_{up} \cdot W_{down}$ (低秩, rank=$r$)。

`★ 核心 Insight`：
**LoRA 是去掉非线性的 Parallel Adapter。**
这是 MAM Adapter 论文最关键的洞察，**所有 PEFT 方法可统一为同一框架**。下下节 (L6) 详讲。

---

## Slide 6: 三方对比（同公式视角）

| 方法 | 公式 | 切入位置 | 非线性 | 可合并 |
|------|------|---------|--------|--------|
| **Pfeiffer (串联)** | $h' + W_{up}\sigma(W_{down}h')$ | base 后 | ✅ | ❌ |
| **Parallel Adapter** | $h' + s \cdot W_{up}\sigma(W_{down}x)$ | base 旁 | ✅ | ❌ |
| **LoRA** | $h' + \frac{\alpha}{r} W_{up}W_{down}x$ | base 旁 | ❌ | ✅ |

**逐步演化**: Pfeiffer → Parallel (改位置) → LoRA (去非线性 + 可合并)

---

## Slide 7: 代码（minimal）

```python
class ParallelAdapter(nn.Module):
    def __init__(self, base_module, r=16, scaling=1.0):
        super().__init__()
        for p in base_module.parameters():
            p.requires_grad = False
        self.base = base_module

        # 推断 in/out 维度
        # GPT-2 MLP: d_in = d_out = 768

        self.down = nn.Linear(d_in, r)
        self.up = nn.Linear(r, d_out)
        self.act = nn.GELU()
        self.scaling = scaling

        nn.init.zeros_(self.up.weight)  # 零初始化
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        base_out = self.base(x)
        adapter_out = self.up(self.act(self.down(x))) * self.scaling
        return base_out + adapter_out
```

**注意**: `down` 的输入是 $x$（原始输入），不是 `base(x)`（这是与串联的关键差异）。

---

## Slide 8: 参数量对比

Parallel Adapter vs Pfeiffer **完全相同**（GPT-2 d=768 时）:

| 方法 | 参数布局 | 总数 |
|------|---------|------|
| Pfeiffer | down(768→16) + up(16→768) per layer | 304K |
| Parallel | down(768→16) + up(16→768) per layer | **304K** ← 一样 |

→ **形状一样，但训练后权重不同**（因为结构不同导致梯度不同）。

测试 `test_serial_vs_parallel_diff` 中验证：训练 5 步后两者 down 权重差异 ~3.3e-3。

---

## Slide 9: 实验 claim（论文）

He et al. 在 XSum 摘要任务上:

| 方法 | ROUGE-L | 参数量 |
|------|---------|--------|
| Pfeiffer | 35.2 | 894K |
| **Parallel Adapter** | **35.6** | 894K |
| LoRA | 35.1 | 884K |
| MAM (混合) | **36.4** | 1.78M |

**关键 takeaway**:
- Parallel 略优于串联（+0.4）
- LoRA 略劣于 Parallel（-0.5）
- 但都不如混搭 MAM（下节）

---

## Slide 10: 并联的工程优势

1. **GPU 并行度高**: base 和 adapter 同时算
2. **梯度路径短**: 直接 $\partial \mathcal{L} / \partial W_{down}$ 不经过 base
3. **结构清晰**: residual 思想更明显

**劣势**:
- 计算量略多（要算 adapter(x) 而非 adapter(base(x))，输入维度可能更大）
- 仍有非线性，不能合并到 base

---

## Slide 11: 思考题

**公式题**:
1. 推导 Parallel Adapter 对 $W_{down}$ 的梯度，与 Pfeiffer 比较，哪个梯度信号更强？
2. 证明：当 $\sigma$ 是恒等映射时，Parallel Adapter 退化为 LoRA。
3. 假设 $\sigma = \tanh$ 且 $W_{down} x$ 很小，证明 Parallel ≈ LoRA。

**设计题**:
4. 把 Parallel Adapter 用在 attention 上 vs FFN 上，哪个效果更好？为什么？
5. 如果 $s = 1$ vs $s = 4$，怎么调？

**对比题**:
6. 用一张图把 Pfeiffer/Parallel/LoRA 三者画在同一个 transformer block 中。

---

## Slide 12: 与下节衔接 — (IA)³

**Adapter 思想的极限**: 我们一路压缩参数:
- Houlsby: 600K (per GPT-2)
- Pfeiffer: 304K
- Compacter: 83K
- **(IA)³ (下节)**: **0.025K** (12 × 2 + 12 × 768 = 9K，三个对角向量)

**(IA)³ 思想**: 把所有矩阵乘换成 element-wise 缩放向量。
$$h' = h \odot \ell$$

→ 参数极少但效果意外好。下节细讲。

---

## Slide 13: 本节小结

```
┌──────────────────────────────────────────────┐
│ Parallel Adapter                             │
│   公式: base(x) + s · up(σ(down(x)))         │
│   关键: down 输入是 x，不是 base(x)          │
│   参数量: 同 Pfeiffer                        │
│   性能: 略优于串联 (+0.4 ROUGE-L)            │
│                                              │
│   核心洞察: 去掉 σ → LoRA                    │
│   (为 L6 MAM Adapter 铺路)                   │
└──────────────────────────────────────────────┘
              ↓
         下节: (IA)³ (极致压缩)
```
