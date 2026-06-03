# Lecture 2: AdapterFusion

> 论文: Pfeiffer et al. 2021, "AdapterFusion: Non-Destructive Task Composition for Transfer Learning" (EACL)
> 配套代码: `src/adapterfusion_minimal.py` + `src/adapterfusion_adapters.py`
> 配套 notebook: `notebooks/02-adapterfusion.ipynb`

---

## Slide 1: 本节路线

```
痛点 → 多任务组合 → 两阶段训练 → Fusion attention 公式 → 代码 → 思考
```

**学完本节，你应该能**:
1. 解释 multi-task learning 的"灾难性遗忘"
2. 写出 Fusion attention 公式
3. 复现两阶段训练流程

---

## Slide 2: 痛点 — 多任务怎么组合

**场景**: 你有 SST-2 (情感)、CoLA (语法)、QQP (问答) 三个任务的 adapter，现在来了第四个 SNLI (蕴含) 任务。

**方案 A**: 全部联合训练 → 灾难性遗忘 + 任务冲突
**方案 B**: 串行 → 后训练破坏前训练（continual learning 难题）
**方案 C**: 平均所有 adapter 输出 → 不灵活，不能适应 token-level 差异

**AdapterFusion**: 训完每个 adapter 后**冻结**，新任务上**学一个 fusion 层**决定如何组合。

---

## Slide 3: 两阶段训练流程

```
Stage 1: Knowledge Extraction (单独训每个 task adapter)
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │ SST-2 data  │ │ CoLA data   │ │ QQP data    │
  │     ↓       │ │     ↓       │ │     ↓       │
  │ Pfeiffer    │ │ Pfeiffer    │ │ Pfeiffer    │
  │ adapter A   │ │ adapter B   │ │ adapter C   │
  └─────────────┘ └─────────────┘ └─────────────┘

Stage 2: Knowledge Composition (冻 adapter，训 fusion)
  ┌──────────────────────────────────────────────┐
  │  New task data (e.g., SNLI)                  │
  │              ↓                               │
  │  Frozen GPT-2 + {Frozen A, B, C}             │
  │              ↓                               │
  │      Fusion attention (Trainable)            │
  │              ↓                               │
  │           SNLI loss                          │
  └──────────────────────────────────────────────┘
```

**关键**: Stage 2 时所有 base + adapter 都冻结，**只训 fusion**。

---

## Slide 4: Fusion attention 公式

每个 transformer block 内的 fusion 层:

输入:
- $h \in \mathbb{R}^{d}$: 当前 hidden（来自 FFN 输出）
- $\{a_1(h), a_2(h), ..., a_N(h)\}$: N 个冻结 adapter 的输出

计算:
$$
Q = W_q \cdot h, \quad K_i = W_k \cdot a_i(h), \quad V_i = W_v \cdot a_i(h)
$$
$$
\alpha_i = \text{softmax}_i\left(\frac{Q \cdot K_i^T}{\sqrt{d}}\right)
$$
$$
\text{fused} = \sum_{i=1}^N \alpha_i \cdot V_i
$$

**直观**: hidden 自己"投票"决定该用哪个 adapter 知识。

---

## Slide 5: 参数量分析（GPT-2, N=3, d=768）

**单 block fusion 参数**:
- $W_q, W_k, W_v$ 各 $d \times d = 768^2 = 589{,}824$
- 三个矩阵合计 $3 \times 589{,}824 = 1{,}769{,}472$

**12 block 合计**: $1{,}769{,}472 \times 12 = 21{,}233{,}664 \approx 21M$

`★ 注意`：21M 远多于 Pfeiffer adapter 本身（304K）。
→ Fusion 层比 adapter 重 ~70 倍。
→ 这是 trade-off：更强的组合能力，但参数膨胀。

---

## Slide 6: 与 attention 机制的等价性

观察公式:
$$\alpha_i = \text{softmax}\left(\frac{Q \cdot K_i^T}{\sqrt{d}}\right), \quad \text{out} = \sum \alpha_i V_i$$

这本质就是 **N-key cross-attention**，其中:
- Query: 当前 hidden ("我现在需要什么")
- Keys: 各 adapter 的输出 ("各 adapter 能提供什么")
- Values: 各 adapter 的输出 ("各 adapter 实际给出的内容")

→ AdapterFusion 是 **"task-level attention"**。

---

## Slide 7: 代码核心（minimal）

```python
class FusionLayer(nn.Module):
    def __init__(self, d, n_adapters):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_v.weight)  # 初始恒等映射

    def forward(self, x, adapter_outs):
        Q = self.W_q(x)
        A = torch.stack(adapter_outs, dim=2)  # (b, s, N, d)
        K = self.W_k(A); V = self.W_v(A)
        scores = (Q.unsqueeze(2) * K).sum(-1) / sqrt(self.d)
        attn = softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * V).sum(dim=2)
```

---

## Slide 8: V 矩阵的恒等初始化

```python
nn.init.eye_(self.W_v.weight)  # V = identity matrix
```

**原因**: 初始时 $V_i = a_i(h)$（没有变换）。

→ 第一步 fusion 输出 ≈ adapter 输出的平均（按 attn 加权），不会引入额外 noise。
→ 与 Adapter 的"up 层零初始化"是同一类思想——**初始扰动最小化**。

---

## Slide 9: 调包对照（adapters 库）

```python
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Fuse

model = AutoAdapterModel.from_pretrained("gpt2")
# 假设已加载 3 个预训练 adapter
for t in ["task_A", "task_B", "task_C"]:
    model.add_adapter(t, config=AdapterConfig.load("pfeiffer"))

# 加 fusion
fusion_setup = Fuse("task_A", "task_B", "task_C")
model.add_adapter_fusion(fusion_setup)
model.train_adapter_fusion(fusion_setup)  # 只训 fusion
```

**实验对照**:
- minimal: 21,233,664 参数
- lib:     21,252,096 参数
- 差异:    0.09%（lib 多了 bias）

→ **弱一致**（实现细节差异）。

---

## Slide 10: AdapterFusion 实验 claim（论文）

Pfeiffer 在 GLUE 上的实验:

| 方法 | GLUE avg | 备注 |
|------|---------|------|
| Pfeiffer adapter (单任务) | 84.2 | baseline |
| Pfeiffer 联合训练 (多任务) | 82.5 | ↓ 灾难性遗忘 |
| **AdapterFusion** | **85.8** | ↑ 1.6 over baseline |

**关键 takeaway**: AdapterFusion **可以系统性地超过单任务训练**，证明"冷冻 + 组合"优于"联合训练"。

---

## Slide 11: 工程优势

| 优势 | 说明 |
|------|------|
| **可重用** | 每个 task adapter 训一次就行，所有下游任务共享 |
| **非破坏** | adapter 永远冻结，不会被新任务"污染" |
| **可组合** | 在 100 个 task 中选 5 个组合到新任务 |
| **并行** | task adapter 可独立训练（分布式友好）|

**典型场景**: 大型 SaaS 平台（每个客户一个 adapter，新客户用 fusion 选择最相关的几个）。

---

## Slide 12: 工程劣势

| 劣势 | 说明 |
|------|------|
| **参数量大** | Fusion 21M 远多于 adapter 本身 |
| **推理慢** | 每个 block 要算 N 个 adapter，再做 attention |
| **训练复杂** | 两阶段、多任务管理（哪个 task 该选哪些 adapter）|

**今天还该用吗**: 仅在"多 task 重用"场景；单任务下直接 LoRA/Pfeiffer。

---

## Slide 13: 思考题

**公式题**:
1. 推导：如果只有 $N=1$ 个 adapter，fusion 退化成什么？参数量？
2. 证明：当 $W_v = I$、$W_q = W_k = 0$ 时，fusion 输出等于 adapter 输出的算术平均。
3. 推导 fusion 中 attn $\alpha_i$ 对 $W_q$ 的梯度。

**设计题**:
4. 假设你有 100 个 task adapter，但每个 block 算 100 次太慢。设计一个"adapter 召回"机制。
5. Fusion 中 V 用 identity 初始化，Q/K 用 normal 初始化。如果 Q/K 也用 identity 初始化会怎样？

**对比题**:
6. AdapterFusion vs AdaMix（下下节将讲）有何本质区别？

---

## Slide 14: 与下节衔接

**当前已学**: Houlsby → Pfeiffer → AdapterFusion
**下节**: 解决两个新痛点
- **AdapterDrop**: Pfeiffer 推理时延 +10%，怎么减？
- **Compacter**: Adapter 参数 ~300K，怎么再压缩 10×？

下节合讲两个 idea（都是"如何让 adapter 更廉价"）。

---

## Slide 15: 本节小结

```
┌─────────────────────────────────────────┐
│ AdapterFusion = "task-level attention"  │
│   Stage 1: 单训每个 adapter (冻 base)   │
│   Stage 2: 冻 adapter, 训 fusion 层     │
│   公式: softmax(Q·K^T) · V              │
│   参数: 12 × 3 × d² ≈ 21M (per block)   │
└─────────────────────────────────────────┘
                    ↓
        非破坏 + 可重用 + 可组合
                    ↓
              缺点: 参数膨胀
                    ↓
             下节: 如何让 adapter 更廉价
```
