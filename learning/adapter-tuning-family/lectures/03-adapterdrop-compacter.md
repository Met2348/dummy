# Lecture 3: AdapterDrop + Compacter

> 论文 1: Rücklé et al. 2020, "AdapterDrop: On the Efficiency of Adapters" (EMNLP)
> 论文 2: Karimi Mahabadi et al. 2021, "Compacter: Efficient Low-Rank Hypercomplex Adapter Layers" (NeurIPS)
> 配套代码: `src/adapterdrop_minimal.py` + `src/compacter_minimal.py`
> 配套 notebook: `notebooks/03-adapterdrop-compacter.ipynb`

---

## Slide 1: 本节路线

```
Adapter 太慢/太大 → 两种解决方案 → AdapterDrop (推理加速) → Compacter (参数压缩) → 实验
```

**学完本节，你应该能**:
1. 实施推理时 adapter 选择性丢弃
2. 推导 PHM (Parameterized Hypercomplex Multiplication) 数学
3. 算清 Compacter 的压缩比

---

## Slide 2: 两个痛点

**Houlsby/Pfeiffer 的问题**:

| 维度 | 问题 |
|------|------|
| **推理速度** | adapter 加 5-10% 时延，多 task 时累计 |
| **参数量** | 单 adapter 25K, GPT-2 共 300-600K，对超大模型仍可观 |

**两个解决方案**:
- **AdapterDrop** (2020): 推理时丢前 k 层 → 加速
- **Compacter** (2021): 用 PHM 数学压缩 → 减参

两者正交，可同时用。

---

## Slide 3: AdapterDrop — 直观

```
观察 (Rücklé et al.):
  对 GLUE 等任务，浅层 adapter 的贡献小于深层
  → 浅层 adapter 可以"白嫖"剪掉

策略:
  ┌──────────────┐
  │ Layer 0      │  ← 丢
  │ Layer 1      │  ← 丢
  │ Layer 2      │  ← 丢
  │ Layer 3      │  ← 丢
  │ Layer 4      │  ← 丢   (k=5)
  ├──────────────┤
  │ Layer 5      │  ← 保留
  │ Layer 6-11   │  ← 保留
  └──────────────┘

效果: 推理时延降低 ~25%（适用于 k=5）
```

---

## Slide 4: AdapterDrop — 训练策略

**Standard AdapterDrop** (训练时也随机丢):
```python
during training:
  for layer in transformer:
    if random() < drop_prob:
      skip_adapter(layer)
    else:
      apply_adapter(layer)
```

**好处**: 模型在训练时就习惯了"部分 adapter 缺失"，推理时丢更鲁棒。

**类比**: 与 Dropout 相同思想（随机 mask + 推理时不 mask）。

---

## Slide 5: AdapterDrop 代码

```python
class _DroppableMlpWrapper(nn.Module):
    def __init__(self, base_mlp, adapter, layer_idx):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = adapter
        self.layer_idx = layer_idx
        self.drop_prob = 0.0           # 训练时丢概率
        self.permanent_drop = False    # 推理时永久丢

    def forward(self, x):
        h = self.base_mlp(x)
        if self.permanent_drop:
            return h
        if self.training and random() < self.drop_prob:
            return h
        return self.adapter(h)
```

`model.set_inference_drop(k=5)` → 把前 5 层 `permanent_drop=True`。

---

## Slide 6: Compacter — 痛点

**Pfeiffer adapter (r=16)**:
- per layer: 2 × 16 × 768 + bias ≈ 25K
- GPT-2 12 层: 300K

**问题**: 对 LLaMA-70B (d=8192) 这种大模型，单 adapter 就 ≈ 300K，12 层 ≈ 3.6M。需要更激进的压缩。

**Compacter 思想**: 把 adapter 的 down/up 矩阵**用 PHM 表示**，可压缩到 1/n。

---

## Slide 7: PHM — 数学定义

**PHM (Parameterized Hypercomplex Multiplication)**:

把 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 分解为 $n$ 个 Kronecker 积之和:
$$W = \sum_{i=1}^{n} A_i \otimes B_i$$

其中:
- $A_i \in \mathbb{R}^{n \times n}$ — 小的"形状矩阵"
- $B_i \in \mathbb{R}^{d_{out}/n \times d_{in}/n}$ — 大的"内容矩阵"

**约束**: $n | d_{out}$ 且 $n | d_{in}$（整除）。

---

## Slide 8: Kronecker 积回顾

$$
\underbrace{\begin{bmatrix} a & b \\ c & d \end{bmatrix}}_{A} \otimes \underbrace{\begin{bmatrix} e & f \\ g & h \end{bmatrix}}_{B} = \begin{bmatrix} aE & bE & aF & bF \\ cE & dE & cF & dF \\ aG & bG & aH & bH \\ cG & dG & cH & dH \end{bmatrix}
$$

等价定义: $(A \otimes B)_{ip+r, jq+s} = A_{ij} \cdot B_{rs}$

**直观**: $A$ 是"宏观结构"，$B$ 是"微观结构"，复合得到完整 $W$。

`★ Insight`：Kronecker 积是 LoHa 的 "LoKr" 同名兄弟。**LoKr 实际上就是 LoRA + PHM!**

---

## Slide 9: 参数量计算

**普通 Linear**: $d_{out} \times d_{in}$

**PHM with n decompositions**:
- $n$ 个 $A_i$: $n \times n^2 = n^3$
- $n$ 个 $B_i$: $n \times (d_{out}/n) \times (d_{in}/n) = d_{out} d_{in}/n$

**总计**: $n^3 + d_{out} d_{in} / n$

**压缩比** (当 $d_{out} d_{in} \gg n^4$): $\approx n$

---

## Slide 10: 具体例子（d=768, r=16, n=4）

**Pfeiffer adapter per layer**:
- down: $768 \times 16 = 12{,}288$ + bias $16$ = $12{,}304$
- up:   $16 \times 768 = 12{,}288$ + bias $768$ = $13{,}056$
- 合计: $25{,}360$

**Compacter adapter per layer**:
- down PHM: $4^3 + 768 \times 16 / 4 = 64 + 3072 = 3{,}136$
- up PHM:   $4^3 + 16 \times 768 / 4 = 64 + 3072 = 3{,}136$
- bias: $16 + 768 = 784$
- 合计: $7{,}056$

**单层压缩比**: $25{,}360 / 7{,}056 = 3.59$×

---

## Slide 11: 跨层共享（关键创新）

**Compacter 进一步**: 所有 12 层共享同一组 $A_i$。

**节省**: 12 × 128 - 128 = 1,408 个参数（小但思想很重要）。

```python
class CompacterGPT2(nn.Module):
    def __init__(self):
        # 创建一次性的 shared A 矩阵
        self.shared_A_down = nn.Parameter(torch.empty(n, n, n))
        self.shared_A_up = nn.Parameter(torch.empty(n, n, n))

        # 每个 block 复用 shared_A
        for block in self.lm.transformer.h:
            adapter = CompacterAdapter(
                d, r, n=n,
                shared_A_down=self.shared_A_down,  # 共享!
                shared_A_up=self.shared_A_up,
            )
```

---

## Slide 12: Compacter 总参数量

**最终统计**（GPT-2, d=768, r=16, n=4）:
- shared A_down: 64
- shared A_up:   64
- per layer (B + bias): 3072 × 2 + 784 = 6,928
- 12 层合计: 64 × 2 + 12 × 6,928 = **83,264**

**与 Pfeiffer 对比**:
- Pfeiffer: 304,320
- Compacter: 83,264
- **压缩比**: 3.65×

---

## Slide 13: Compacter — 反向梯度

对 $B_i$ 的梯度（标准矩阵乘法链规则）:
$$\frac{\partial \mathcal{L}}{\partial B_i} = \frac{\partial \mathcal{L}}{\partial W} \cdot \frac{\partial (A_i \otimes B_i)}{\partial B_i}$$

由于 Kronecker 积是 $A$ 和 $B$ 的双线性映射，对 $B$ 求导本质上是"按 $A_{jk}$ 的权重"分发梯度到 $B$ 的对应块。

**实现细节**: 不需要手写，autograd 自动处理。

---

## Slide 14: 代码核心（PHMLinear）

```python
class PHMLinear(nn.Module):
    def __init__(self, in_features, out_features, n=4, shared_A=None):
        super().__init__()
        if shared_A is not None:
            self.A = shared_A
        else:
            self.A = nn.Parameter(torch.empty(n, n, n))
            nn.init.kaiming_uniform_(self.A, a=sqrt(5))
        self.B = nn.Parameter(torch.empty(n, out_features // n, in_features // n))
        nn.init.kaiming_uniform_(self.B, a=sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def construct_weight(self):
        return sum(kronecker(self.A[i], self.B[i]) for i in range(self.n))

    def forward(self, x):
        return x @ self.construct_weight().T + self.bias
```

---

## Slide 15: 与 adapters 库对比

```python
from adapters import AutoAdapterModel, AdapterConfig
config = AdapterConfig.load("compacter", reduction_factor=48, phm_dim=4)
model.add_adapter("demo", config=config)
```

**实验对比**:
- minimal: 83,264 参数
- adapters lib: 56,512 参数（无 bias / 不同共享 scheme）
- 差异 ~32%

→ **弱一致**（不同库的 Compacter 实现细节不同；论文版本被多次修订）。

---

## Slide 16: 实验 — 论文 claim

Compacter 在 GLUE 上的实验:

| 方法 | 参数量 | GLUE avg |
|------|--------|---------|
| Pfeiffer adapter | 894K (BERT-base) | 86.5 |
| **Compacter** | **64K** (BERT-base) | **86.4** |

**关键 takeaway**:
- 参数压缩 **14×**
- 性能几乎相同 (-0.1)

→ PHM 的"结构化压缩"是有效的低秩归纳偏置。

---

## Slide 17: 思考题

**公式题**:
1. 写出 Kronecker 积的两个性质: $(A \otimes B)(C \otimes D) = ?$ 和 $\text{tr}(A \otimes B) = ?$
2. 推导 PHM 在 $n=1$ 时退化成什么形式。$n=d$ 时呢？
3. 算 GPT-2 (d=768) 在 $n=8$ 时的参数量与 $n=4$ 时哪个更小？

**设计题**:
4. 你怎么决定 $n$ 的最优值？（提示: 性能 vs 参数量 trade-off）
5. AdapterDrop 训练时随机丢 vs 推理时永久丢，分别等价于什么经典正则？
6. 假设你要做 LLaMA-70B 的 Compacter，$d=8192$，$n=8$，per layer 参数量？

**对比题**:
7. Compacter (PHM) vs LoKr (LoRA + Kronecker) 是不是一回事？

---

## Slide 18: 工程选型

| 情况 | 推荐 |
|------|------|
| 普通 PEFT（d<1024）| 直接 Pfeiffer/LoRA |
| 大模型 (d > 4096) + 多 task | Compacter（参数压缩有意义）|
| 推理时延敏感 + 多 task | AdapterDrop |
| 极致省参 (< 0.01%) | (IA)³（下下节）|

**今天工业界**: Compacter 不常用（LoRA + 量化更省），但学习 PHM 数学有价值。

---

## Slide 19: 下节预告 — Parallel Adapter

**到此为止**: 所有 adapter 都是 **串联** (sequential) 结构。
$$h \to \text{base} \to \text{adapter} \to h'$$

**问题**: 串联意味着 adapter 必须等 base 算完。能否**并联** (parallel)？
$$h \to \begin{cases} \text{base}(h) \\ \text{adapter}(h) \end{cases} \to h + \text{base}(h) + \text{adapter}(h)$$

→ He et al. 在 "Towards a Unified View" 提出 Parallel Adapter。

下节细讲（并联 ≈ LoRA 的特殊形式）。

---

## Slide 20: 本节小结

```
┌─────────────────────────────────────────────────┐
│ AdapterDrop = "Dropout for adapters"            │
│   训练: 随机丢          推理: 永久丢前 k 层      │
│   加速 ~25% (k=5)                                │
├─────────────────────────────────────────────────┤
│ Compacter = "PHM for adapter compression"       │
│   W = Σᵢ Aᵢ ⊗ Bᵢ                                │
│   shared A across layers                        │
│   参数压缩 3-14× 视场景而定                       │
└─────────────────────────────────────────────────┘
              ↓
       下节: Parallel Adapter
```
