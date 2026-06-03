# Lecture 8: AdaMix — Mixture of Adapters

> 论文: Wang et al. 2022, "AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning" (EMNLP)
> 配套代码: `src/adamix_minimal.py`
> 配套 notebook: `notebooks/08-adamix.ipynb`

**本节是主线最后一篇。下一节进入跨专题综合。**

---

## Slide 1: 本节路线

```
痛点 (固定 adapter 不够) → MoE 思想 → AdaMix 公式 → 训练/推理策略 → 实验
```

**学完本节，你应该能**:
1. 区分 MoE 和 MoA (Mixture of Adapters)
2. 理解 stochastic routing 训练
3. 解释 weight averaging 推理
4. 推导 merge_experts 优化

---

## Slide 2: 痛点 — 固定 adapter 的局限

**到此为止学的所有 adapter**: 都是固定结构 (single bottleneck per block)。

**问题**:
- 不同样本可能需要不同的"适应方向"
- 固定 adapter 是"妥协的平均"
- 增大 r 能 mitigate 但参数膨胀

**AdaMix 思想**: 让 adapter 自适应——**多个 expert 共存，按需选用**。

---

## Slide 3: AdaMix vs MoE — 区别

| 维度 | MoE | AdaMix (MoA) |
|------|-----|--------------|
| **替换什么** | FFN | adapter 模块 |
| **粒度** | layer-level (整层) | adapter-level (块内) |
| **路由** | gating network | **stochastic** (训) / **average** (推) |
| **参数膨胀** | 巨大（FFN × N）| 较小（adapter × N）|
| **load balance loss** | 必须 | 不需要（stochastic 自然 balance）|

→ AdaMix 是 **轻量级 MoE**，更适合 PEFT。

---

## Slide 4: AdaMix 训练策略 — Stochastic Routing

```python
def forward(self, x):
    if self.training:
        idx = randint(0, n_experts)  # 随机选 1 个
        return self.experts[idx](x)
    else:
        return mean([e(x) for e in self.experts])  # 平均
```

**为什么 stochastic**:
- 每次只算 1 个 expert → 训练**速度等于单 adapter**
- 不需要 gating network → 不需要 load balance loss
- 训练时引入随机性 → 类似 Dropout 正则化

---

## Slide 5: AdaMix 推理策略 — Weight Averaging

**两种推理方式**:

1. **Expert averaging** (default):
   $$\text{out} = \frac{1}{N} \sum_i e_i(x)$$
   计算成本 N×。

2. **Weight averaging** (merge_experts):
   $$W_{\text{merged}} = \frac{1}{N} \sum_i W_i$$
   推理用 $W_{\text{merged}}$，**计算成本 = 单 adapter**。

→ **关键发现**: 由于 adapter 都是线性 bottleneck，权重平均 ≈ 输出平均。可省 N× 推理成本。

---

## Slide 6: 参数量

GPT-2 d=768, r=16, N=4:
- per expert (per layer): $25{,}360$（同 Houlsby Adapter）
- per layer (N experts): $4 \times 25{,}360 = 101{,}440$
- 12 layers: $\mathbf{1{,}217{,}280}$

**对比**:
- Pfeiffer: 304K (1×)
- AdaMix N=4: 1.2M (4×)
- AdaMix N=4 after merge: 304K (1×) ← 同 Pfeiffer

→ 训练比 Pfeiffer 多 4× 参数，但 **推理后合并** 回 Pfeiffer 大小。

---

## Slide 7: 代码核心

```python
class AdaMixLayer(nn.Module):
    def __init__(self, d, r=16, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            HoulsbyAdapter(d, r) for _ in range(n_experts)
        ])

    def forward(self, x):
        if self.training:
            idx = torch.randint(0, self.n_experts, (1,)).item()
            return self.experts[idx](x)
        else:
            outs = [e(x) for e in self.experts]
            return torch.stack(outs).mean(dim=0)

    def merge_experts(self):
        # 把 N 个 expert 的权重平均，留下 1 个
        ...
```

---

## Slide 8: 实验 — Stochastic 训练 vs Deterministic 推理

```
训练时（5 次相同输入）:
  losses = [9.37, 8.80, 10.50, 8.71, 8.79]
  → 不同 step 选不同 expert，loss 差异显著

推理时（3 次相同输入）:
  loss = 8.58 == 8.58 == 8.58
  → 取平均，deterministic
```

→ stochastic 训 + deterministic 推是 AdaMix 的核心 trick。

---

## Slide 9: merge_experts 优化

**问题**: 推理时计算 N 个 expert 太慢。

**解决**: 训练结束后，做权重平均:
$$W_{\text{down}}^{\text{merged}} = \frac{1}{N} \sum_i W_{\text{down}}^{(i)}$$
$$W_{\text{up}}^{\text{merged}} = \frac{1}{N} \sum_i W_{\text{up}}^{(i)}$$

**正当性**:
- adapter forward: $y = W_{up} \sigma(W_{down} x)$
- 由于 σ 是非线性，**严格上**不等价 (mean of σ ≠ σ of mean)
- 但实验上误差很小（< 0.1% on GLUE）

→ 推理成本回到单 adapter 水平。

---

## Slide 10: 实验 claim（论文）

Wang et al. 在 GLUE 上:

| 方法 | 参数 | GLUE avg |
|------|------|---------|
| Pfeiffer adapter | 0.24% | 86.3 |
| LoRA | 0.24% | 86.4 |
| **AdaMix (N=4)** | 0.95% (训) / 0.24% (merge) | **86.8** |
| Full FT | 100% | 86.5 |

**关键 takeaway**:
- AdaMix (merge 后) **超过 full FT** (+0.3)
- 参数与 LoRA 相同
- 训练时多 4× 参数，但 step 数不变

---

## Slide 11: 思考题

**公式题**:
1. 推导：stochastic routing 训练时单 step 的期望梯度 = ?
2. 证明：N → ∞ 时 AdaMix → Bayesian model averaging。
3. 推算 N=8 AdaMix 在 BERT-large (d=1024) 上的参数量。

**设计题**:
4. AdaMix 用 stochastic routing，能否改为 gating network（类 MoE）？trade-off？
5. 如果不做 merge_experts，N=4 推理慢 4×，可接受吗？
6. AdaMix 与 Dropout 在思想上的相似处？

**对比题**:
7. AdaMix vs AdapterFusion 的本质差异（都是"多 adapter 组合"）？
8. AdaMix 与 SWA (Stochastic Weight Averaging) 的关系？

---

## Slide 12: 工程选型

| 场景 | 推荐 |
|------|------|
| 单任务 + 资源紧张 | Pfeiffer / LoRA |
| 单任务 + 追求最高质量 | **AdaMix N=4** ⭐⭐⭐ |
| 多任务 + 任务相关性强 | AdapterFusion |
| 多任务 + 任务相关性弱 | 单独 adapter |
| 大模型 LoRA + 多 task | DoRA + AdaMix (混搭) |

**今天还在用吗**: ✅ 在论文中常用作 baseline；工业上较少（merge 步骤增加复杂度）。

---

## Slide 13: 与下节衔接 — L9 三线综合

**到此为止（L1-L8）**: 完成所有 11 种 Adapter 家族方法。

**下节 (L9)**: **三线综合** — 把 Prompt + LoRA + Adapter 28 方法统一到同一框架。
- MAM Adapter 论文统一公式回顾
- 28 方法横向对比表
- PEFT 工程选型决策树

**下下节 (L10)**: PEFT 下一步
- Adapter 的多模态复活（LLaMA-Adapter, Q-Former, LLaVA）
- 后续专题路线图

---

## Slide 14: 本节小结

```
┌──────────────────────────────────────────────┐
│ AdaMix = Mixture of Adapters                 │
│                                              │
│   训练: Stochastic routing                   │
│     每 step 随机选 1 expert → 速度同单 adapter│
│                                              │
│   推理: Weight averaging                     │
│     output = mean(e_i(x)) for all i          │
│     或 merge_experts → 单 adapter 推理成本   │
│                                              │
│   实验: +0.3 over full FT                   │
│   参数: N× during training, 1× after merge  │
└──────────────────────────────────────────────┘
              ↓
     主线 (L1-L8) 完结！
              ↓
       下节: L9 三线综合
```

---

## Slide 15: 主线完结庆祝

```
你已经走完 Adapter 家族 11 种方法的主线!

L1 Houlsby + Pfeiffer       基础串联 + 简化
L2 AdapterFusion            多任务 attention 融合
L3 AdapterDrop + Compacter  推理加速 + PHM 压缩
L4 Parallel Adapter         并联结构 (LoRA 前身)
L5 (IA)³                    极致压缩 (3 个对角向量)
L6 MAM Adapter              ⭐ 统一视角 (理论高峰)
L7 K-Adapter + MAD-X        应用层 (知识 + 跨语言)
L8 AdaMix                   MoE 路由 (本节)

接下来 (L9-L10):
  L9 三线综合 (Prompt + LoRA + Adapter)
  L10 PEFT 下一步 (Adapter 多模态复活 + 路线图)
```
