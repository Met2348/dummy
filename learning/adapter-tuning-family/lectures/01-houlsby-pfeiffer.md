# Lecture 1: Houlsby + Pfeiffer Adapter

> 论文 1: Houlsby et al. 2019, "Parameter-Efficient Transfer Learning for NLP" (ICML)
> 论文 2: Pfeiffer et al. 2020, "AdapterFusion: Non-Destructive Task Composition" (EACL)
> 配套代码: `src/houlsby_minimal.py` + `src/pfeiffer_minimal.py`
> 配套 notebook: `notebooks/01-houlsby-pfeiffer.ipynb`

---

## Slide 1: 本节路线

```
开篇 → Adapter 是什么 → Houlsby 双串联 → Pfeiffer 简化 → 代码 → 实验 → 思考
```

**学完本节，你应该能**：
1. 写出 Adapter 的核心公式
2. 解释为什么 up 层零初始化是关键
3. 推算 Houlsby/Pfeiffer 在 GPT-2 上的参数量
4. 理解 Adapter 与 LoRA 的范式差异

---

## Slide 2: 痛点 — PEFT 的起源（2019）

**2018-2019 年大模型微调的状态**:
- BERT-base (110M) / GPT-2 (124M) 主流
- 全参数微调每个下游任务 → 存储 N × 110M = 海量
- ULMFiT 等"渐进解冻"方法效果不理想

**Houlsby 的洞察**:
> "为什么我们要训练整个模型？也许只需在每个 block 加一个小'补丁'就够。"

由此开启 **PEFT (Parameter-Efficient Fine-Tuning)** 范式。

---

## Slide 3: Adapter 核心公式

$$
\text{Adapter}(x) = \underbrace{x}_{\text{残差}} + W_{up} \cdot \sigma(W_{down} \cdot x)
$$

其中:
- $W_{down} \in \mathbb{R}^{r \times d}$: 降维矩阵（bottleneck）
- $\sigma$: 非线性激活（GELU/ReLU）
- $W_{up} \in \mathbb{R}^{d \times r}$: 升维还原
- $r \ll d$: 通常 $r = 16, d = 768$（GPT-2 base）

**几何直观**: 把高维特征"压缩到 $r$ 维子空间 → 非线性变换 → 还原"。
是 task-specific 的"特征调整器"。

---

## Slide 4: Houlsby 结构图（双串联）

```
                    ┌──────────────────┐
        ┌─────────→ │ Layer Norm       │
        │           └──────────────────┘
        │                    ↓
        │           ┌──────────────────┐
        │           │  Attention       │
        │           └──────────────────┘
        │                    ↓
        │           ┌──────────────────┐  ← 残差 +
        │           │ Adapter (Houlsby)│
        │           └──────────────────┘
        ↓                    ↓ + 残差 (block-level)
        ─────────→  ⊕
                            ↓
        ┌─────────────────────────────┐
        │ Layer Norm                  │
        └─────────────────────────────┘
                            ↓
        ┌─────────────────────────────┐
        │ FFN (MLP)                   │
        └─────────────────────────────┘
                            ↓
        ┌─────────────────────────────┐  ← 第二个 adapter
        │ Adapter (Houlsby)           │
        └─────────────────────────────┘
                            ↓
                    ⊕ (residual)
```

**关键**: 每 block 插 **2 个 adapter**（attention 后 + FFN 后）。

---

## Slide 5: Pfeiffer 简化（每 block 1 个）

```
        block input
            ↓
        Attention + 残差
            ↓                          ← 不插 adapter
        FFN
            ↓
        ┌──────────────┐
        │ Adapter only │              ← 只插这里
        └──────────────┘
            ↓
        + 残差
```

**论文 claim**: 实验上 Pfeiffer ≈ Houlsby 效果，参数减半 → "second adapter is all you need"。

---

## Slide 6: 参数量推导（GPT-2, d=768, r=16）

**单个 Adapter**:
- $W_{down}$: $r \times d = 16 \times 768 = 12{,}288$ + bias $16$ = $12{,}304$
- $W_{up}$:   $d \times r = 768 \times 16 = 12{,}288$ + bias $768$ = $13{,}056$
- 合计: $25{,}360$

**Houlsby (每 block × 2 × 12 block)**:
$25{,}360 \times 24 = 608{,}640$ ≈ 0.49% of base

**Pfeiffer (每 block × 1 × 12 block)**:
$25{,}360 \times 12 = 304{,}320$ ≈ 0.24% of base

**对比 LoRA r=8**: 294,912 ≈ 0.24%（Pfeiffer 几乎一样）

---

## Slide 7: 零初始化的核心作用

**为什么 $W_{up}$ 零初始化？**

初始 forward:
$$\text{Adapter}(x) = x + \underbrace{W_{up}}_{=0} \cdot \sigma(W_{down} \cdot x) = x$$

**好处**:
1. 初始模型行为 = 原始 base model（无扰动）
2. 训练初期不破坏已学好的表示
3. 数值稳定（不会因随机初始化引入 noise）

这与 LoRA 的 $B$ 零初始化是同一种思想，**所有 PEFT 方法都遵循"零初始化扰动"原则**。

---

## Slide 8: 反向梯度

对 $W_{down}$ 的梯度:
$$\frac{\partial \mathcal{L}}{\partial W_{down}} = \frac{\partial \mathcal{L}}{\partial h_{\text{adapter}}} \cdot W_{up}^T \cdot \text{diag}(\sigma'(W_{down}x)) \cdot x^T$$

初始时 $W_{up} = 0$ → $\frac{\partial \mathcal{L}}{\partial W_{down}} = 0$ → **$W_{down}$ 第一步不更新**。

但 $W_{up}$ 第一步**会**更新（不依赖 $W_{up}$）：
$$\frac{\partial \mathcal{L}}{\partial W_{up}} = \frac{\partial \mathcal{L}}{\partial h_{\text{adapter}}} \cdot \sigma(W_{down}x)$$

**收敛规律**: $W_{up}$ 先动，$W_{down}$ 随后被激活，两层交替前进。

---

## Slide 9: Houlsby vs Pfeiffer — 工程对比

| 维度 | Houlsby | Pfeiffer |
|------|---------|----------|
| 每 block adapter 数 | 2 | 1 |
| 参数量 (GPT-2, r=16) | 608K | 304K |
| 性能 (GLUE 平均) | 84.0 | **84.2** |
| 训练时延 | 1.0× | **0.9×** |
| 推理时延 | 1.0× | **0.9×** |

**Pfeiffer 几乎全胜**。所以 2020 后社区主用 Pfeiffer 配置。

---

## Slide 10: Houlsby 代码（minimal 核心 30 行）

```python
class HoulsbyAdapter(nn.Module):
    def __init__(self, d, r=16):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)  # 关键：零初始化
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class _AttnAdapterWrapper(nn.Module):
    """挂在 attn 输出后。"""
    def __init__(self, base_attn, adapter):
        super().__init__()
        self.base_attn = base_attn
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        outputs = self.base_attn(*args, **kwargs)
        outputs = (self.adapter(outputs[0]),) + outputs[1:]
        return outputs
```

---

## Slide 11: 调包对照（adapters 库）

```python
from adapters import AutoAdapterModel, AdapterConfig

model = AutoAdapterModel.from_pretrained("gpt2")
config = AdapterConfig.load("houlsby", reduction_factor=48)
# reduction_factor = d / r：48 = 768 / 16
model.add_adapter("my_task", config=config)
model.train_adapter("my_task")  # 自动冻结 base
```

**adapters 库 vs minimal 参数量对比**:
- Houlsby: 608,640 vs 608,640 ✅ 完美一致
- Pfeiffer: 304,320 vs 304,320 ✅

→ **强一致性**：两个独立实现产出完全相同的参数布局。

---

## Slide 12: 实验 — 初始 forward = base?

`tests/test_houlsby_pfeiffer.py::test_initial_forward` 实测:

```
Houlsby vs base:  max |Δlogits| = 0.0000e+00  ✅
Pfeiffer vs base: max |Δlogits| = 0.0000e+00  ✅
```

**完美 bit 一致**：零初始化的 $W_{up}$ 让 Adapter 透明插入。

---

## Slide 13: 实验 — mini training

`tests/test_houlsby_pfeiffer.py::test_mini_training`:

```
4 样本 × 10 步 × AdamW(lr=1e-3)
loss: 6.952 → 0.885   (Houlsby r=16)
```

收敛正常。如果你跑 notebook 会看到完整曲线。

---

## Slide 14: 与 LoRA 的范式差异

| 维度 | Adapter (Houlsby/Pfeiffer) | LoRA |
|------|---------------------------|------|
| 切入点 | **加层**（结构端）| **改权重**（权重端）|
| 公式 | $h + \text{up}(\sigma(\text{down}(h)))$ | $h + \alpha/r \cdot BAh$ |
| 非线性 | **有** ($\sigma$) | 无 |
| 可合并 | **❌**（非线性阻碍）| ✅ |
| 推理时延 | +5-10% | **0%** |
| 论文年份 | 2019 | 2021 |

**LoRA 本质 = 去掉非线性 + 低秩约束的 Adapter**。
也就是说，**学过 LoRA 再看 Adapter，你已经知道一半**。

---

## Slide 15: 与 Prompt Tuning 的差异

| 维度 | Adapter | Prompt Tuning |
|------|---------|---------------|
| 切入点 | 结构端 | **输入端** |
| 参数位置 | 每 block 内部 | 仅 input embedding |
| context 占用 | 0 | 占 m 个 token 位置 |
| 参数量 | 600K (Houlsby) | 7.7K (T5-base) |
| 灵活性 | 高（可控制每层） | 低（只能影响 input）|

**Prompt 极致省参，Adapter 表达力强**。

---

## Slide 16: 思考题

**公式题**:
1. 推导：如果把 $\sigma$ 改成恒等映射，Adapter 退化成什么形式？
2. 写出 Houlsby Adapter 对 $W_{up}, W_{down}$ 的反向梯度。
3. 证明：当 $\sigma = \tanh$ 且初始化使输入很小时，Adapter ≈ LoRA。

**设计题**:
4. 如果只能保留一个 adapter（attn 后 vs FFN 后），你选哪个？为什么？
5. Pfeiffer 论文 claim FFN 后比 attn 后更重要，你认为原因是什么？

**对比题**:
6. 用一句话说出 Adapter / LoRA / Prefix Tuning 的最根本差异。

---

## Slide 17: 工程选型

**今天还该用 Houlsby/Pfeiffer 吗？**

- ❌ 不建议（除非有特殊需求）：LoRA 几乎全方位更好（参数量、推理时延、效果）
- ✅ 仍有的场景：
  - **多任务 hub**（AdapterHub）—— 几千个预训练 adapter 可即插即用
  - **跨语言 transfer**（MAD-X 用 adapter）—— 多 adapter 组合优势明显
  - **教学**（理解 PEFT 演化）

---

## Slide 18: 下节预告 — AdapterFusion

**问题**: 训了 10 个 task-specific adapter，如何**组合**它们到一个新任务？

**朴素方案**: 平均所有 adapter 输出 → 不够灵活
**Houlsby 方案**: 多任务联合训 → 容易"灾难性遗忘"

**AdapterFusion 方案**:
1. 单独训每个 task adapter（冻 base）
2. 冻所有 task adapter，加 attention 融合层
3. 新任务上学融合权重

下节细讲。

---

## Slide 19: 本节小结

```
开篇  →  ┌────────────────────────────────┐
         │ 2019: Adapter 范式诞生         │
         │ 公式: x + up(σ(down(x)))       │
         │ 关键: up 零初始化              │
         │ 参数量: ~0.5% of base          │
         └────────────────────────────────┘
                         ↓
         ┌────────────────────────────────┐
         │ Houlsby (2 个/block)           │
         │ Pfeiffer (1 个/block)          │
         │ 实验上 Pfeiffer 更优           │
         └────────────────────────────────┘
                         ↓
                   下节: AdapterFusion
```

**关键 takeaway**:
- Adapter = "把 task 知识压缩到 r 维 bottleneck"
- 零初始化是 PEFT 通用范式
- Pfeiffer = Houlsby 简化版，工程默认选 Pfeiffer
