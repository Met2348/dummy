# Lecture 6: MAM Adapter (Mix-And-Match) — 统一视角

> 论文: He et al. 2022, "Towards a Unified View of Parameter-Efficient Transfer Learning" (ICLR)
> 配套代码: `src/mam_minimal.py`
> 配套 notebook: `notebooks/06-mam-adapter.ipynb`

**本节是 Adapter 专题的理论高峰。如果你只能选一节深读，选这一节。**

---

## Slide 1: 本节路线

```
统一框架 → 三大主线在数学上等价 → MAM 最优混搭 → 实验 → 思考
```

**学完本节，你应该能**:
1. 写出 He et al. 的统一公式
2. 证明 Prefix Tuning ≡ Parallel Adapter
3. 证明 LoRA ≡ 去非线性 Adapter
4. 解释 MAM 为什么是最优混合

---

## Slide 2: 痛点 — PEFT 方法这么多，是不是同一回事？

到 2022 年，PEFT 方法井喷:
- Adapter (Houlsby, Pfeiffer, Compacter)
- Prompt Tuning, Prefix Tuning, P-Tuning
- LoRA, (IA)³

**He et al. 的疑问**:
> "这些方法看似不同，但实际上是不是同一种东西的不同伪装？"

答案: **是的**。他们都可以写成同一个公式。

---

## Slide 3: 统一公式（核心定理）

对 Transformer 中任意一个 functional unit（attention 或 FFN）：

$$h \leftarrow h + \Delta h$$

其中 $\Delta h$ 都可以写成:

$$\Delta h = \text{func}(W_{down} \cdot x) \cdot W_{up}$$

只是 **func / down 矩阵 / 应用位置** 不同:

| 方法 | $W_{down}$ | func | $W_{up}$ | 位置 |
|------|-----------|------|----------|------|
| Adapter (Houlsby) | $(r, d)$ | $\sigma$ | $(d, r)$ | base 后 |
| Parallel Adapter | $(r, d)$ | $\sigma$ | $(d, r)$ | 与 base 并联 |
| LoRA | $(r, d)$ | id | $(d, r)$ | 与 base 并联 |
| Prefix Tuning | $(l, d)$ (prefix vecs) | attention | (隐式) | 注入 K, V |
| (IA)³ | diag(d) | id | id | element-wise |

---

## Slide 4: 证明 1 — Prefix Tuning ≡ Parallel Adapter

**Prefix Tuning**: 在 K, V 前拼接 $l$ 个 learnable vectors $P_k, P_v$:
$$K' = [P_k; K], \quad V' = [P_v; V]$$

attention 输出:
$$\text{attn}(Q, K', V') = \text{softmax}\left(\frac{Q K'^T}{\sqrt{d}}\right) V'$$

**He et al. 的关键观察**: 展开 softmax 后，可以分解为:
$$\text{attn}(Q, K', V') = (1 - \lambda(Q)) \cdot \text{attn}(Q, K, V) + \lambda(Q) \cdot \text{attn}(Q, P_k, P_v)$$

其中 $\lambda(Q) = \text{softmax}(Q P_k^T) / [\text{softmax}(Q P_k^T) + \text{softmax}(Q K^T)]$.

第二项是"对原 attention 输出的扰动"，**等价于 Parallel Adapter** 注入。

---

## Slide 5: 证明 2 — LoRA ≡ Parallel Adapter（去非线性）

**Parallel Adapter**:
$$\Delta h = W_{up} \sigma(W_{down} x)$$

**当 $\sigma = \text{identity}$（去非线性）**:
$$\Delta h = W_{up} W_{down} x = \Delta W \cdot x$$

其中 $\Delta W = W_{up} W_{down}$ 是低秩矩阵（rank $\le r$）。

这就是 **LoRA**！

→ **LoRA = 去非线性的 Parallel Adapter**。

---

## Slide 6: 统一视角图

```
┌─────────────────────────────────────────────────┐
│  统一公式: h ← h + f(W_down · x) · W_up         │
│                                                 │
│  Adapter (串联) ┐                               │
│  Parallel       │── f 是非线性                  │
│  Compacter      │                               │
│                 ┘                               │
│  LoRA           ── f 是 identity                │
│                                                 │
│  Prefix         ── 通过 attention 注入          │
│                    (数学等价于 Parallel)        │
│                                                 │
│  (IA)³          ── W_down/W_up 退化为对角阵     │
└─────────────────────────────────────────────────┘
```

---

## Slide 7: MAM (Mix-And-Match) — 最优组合

He et al. 通过实验证明:

| 组件 | 最佳放置 |
|------|---------|
| **Prefix-style** | **Attention 端** |
| **Parallel Adapter** | **FFN 端** |
| (Series/串联 Adapter) | 哪都不放 |
| LoRA | 不如 Parallel |

→ **MAM = Prefix attn + Parallel FFN**

直觉:
- attention 需要"动态"调整（与 query 相关）→ Prefix
- FFN 需要"静态"特征变换 → Parallel Adapter

---

## Slide 8: MAM 公式

每个 transformer block:

$$
\text{Attn output} = \text{Attention}(Q, [P_k; K], [P_v; V])
$$
$$
\text{FFN output} = \text{FFN}(h) + s \cdot W_{up} \sigma(W_{down} h)
$$

参数 per layer (GPT-2 d=768, l=30, r=16):
- Prefix: $2 \times 30 \times 768 = 46{,}080$
- Parallel adapter: $25{,}360$
- 合计: $71{,}440$

12 层: $857{,}280$ ≈ 0.68% of base

---

## Slide 9: MAM 在 XSum 实验

He et al. 在 XSum 摘要任务:

| 方法 | 参数 | ROUGE-L | 性能 |
|------|------|---------|------|
| Pfeiffer | 0.84% | 35.2 | baseline |
| LoRA | 0.84% | 35.1 | -0.1 |
| Parallel | 0.84% | 35.6 | +0.4 |
| Prefix Tuning | 3.6% | 35.7 | +0.5 |
| **MAM Adapter** | **6.7%** | **36.4** | **+1.2** ⭐ |

**关键 takeaway**: MAM 显著优于单一方法，证明"混搭最优"。

---

## Slide 10: 代码核心

```python
class MAMGPT2(nn.Module):
    def __init__(self, prefix_len=30, r=16, scaling=4.0):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        for block in self.lm.transformer.h:
            # 1. Prefix attention
            block.attn.c_attn = PrefixAttention(block.attn.c_attn, d, prefix_len)
            # 2. Parallel FFN
            block.mlp = ParallelAdapter(block.mlp, r=r, scaling=scaling)
```

→ 复用 L4 的 ParallelAdapter + 新写 PrefixAttention。
→ **MAM = Prefix + Parallel 的简单拼装**。

---

## Slide 11: 与 LoRA 系列 DoRA 的关系

`★ 跨专题 Insight`：DoRA (L8 of LoRA family) 与 MAM 都是"组合主义"。
- DoRA = LoRA + 权重分解 (magnitude × direction)
- MAM = Prefix + Parallel

→ 都告诉我们：单一 PEFT 方法的天花板有限，**组合才能突破**。

但 DoRA 仍然可合并，MAM 不可合并（有非线性的 Parallel + 不可合并的 Prefix）。

---

## Slide 12: 统一公式表（28 方法全 PEFT）

| 主线 | 方法 | $f$ | $W_{down}$ | $W_{up}$ |
|------|------|-----|-----------|----------|
| **Prompt** | Prompt Tuning | embedding | (空, 输入端) | (空) |
| | Prefix Tuning | softmax(QK)V | (l, d) | (d, l) |
| | P-Tuning v2 | softmax(QK)V | per-layer prefix | per-layer |
| **LoRA 系** | LoRA | identity | (r, d) | (d, r) |
| | AdaLoRA | identity | $P\Lambda$ | $Q^T$ |
| | PiSSA | identity | SVD init | SVD init |
| | DoRA | identity + norm | (r, d) | $m \cdot (d, r)$ |
| **Adapter** | Houlsby | $\sigma$ | (r, d) | (d, r) |
| | Compacter | $\sigma$ | PHM | PHM |
| | Parallel | $\sigma$ | (r, d) | (d, r) |
| | (IA)³ | identity | diag | identity |
| **混合** | **MAM** | $\sigma$ + softmax | mixed | mixed |

---

## Slide 13: 反向梯度（MAM 两部分）

对 prefix $P_k, P_v$ 的梯度: 通过 attention 链规则
$$\frac{\partial \mathcal{L}}{\partial P_k} = \frac{\partial \mathcal{L}}{\partial \text{attn}} \cdot \frac{\partial \text{attn}}{\partial K'} \cdot [\mathbf{1}_l, 0]$$

对 Parallel adapter $W_{down}, W_{up}$: 与 LoRA 几乎相同（仅多了 $\sigma'$）。

→ 两部分**梯度独立**，因为 attention 和 FFN 在前向上不共享参数。

---

## Slide 14: 思考题

**公式题**:
1. 证明：当 prefix_len $l = 0$，MAM 退化为 Parallel Adapter。
2. 证明：当 r = 0，MAM 退化为 Prefix Tuning。
3. 推导 $\lambda(Q)$ 在何时趋近于 0（Prefix 几乎无效果）？

**设计题**:
4. 如果 attention 端用 LoRA 而不是 Prefix 会怎样？
5. 在 MAM 上加 LoRA 到 attention 端（成为 "MAM + LoRA"）会更好吗？
6. 如何把 MAM 用到只有 attention 没有 FFN 的架构（如 Transformer 变种）？

**对比题**:
7. MAM vs UniPELT（下下下节）有何区别？
8. 用一句话说明"统一视角"的核心洞察。

---

## Slide 15: 工程选型

| 场景 | MAM 适合吗 |
|------|---------|
| 资源紧张 + 极致省参 | ❌ (用 (IA)³) |
| 推理速度敏感 | ❌ (用 LoRA/DoRA) |
| **追求 SOTA 性能** | ✅ ⭐⭐⭐ |
| 研究/论文 | ✅ (展示混搭威力) |
| 复杂 task (long generation) | ✅ |

**今天工业界**: MAM 不常用（实现复杂、推理慢），但学术上证明了"组合的威力"。
DoRA (2024) 在某种意义上继承了 MAM 的"组合主义"思想。

---

## Slide 16: 与下节衔接

**到此为止**: 完成所有"PEFT 理论核心"。

**下节 L7**: 应用层 PEFT
- **K-Adapter**: 注入外部知识到 PLM
- **MAD-X**: 跨语言 transfer（lang + task adapter）

**下下节 L8**: AdaMix (MoE 路由)
**下下下节 L9**: 跨专题三线综合
**最后 L10**: PEFT 的"下一步"（含 Adapter 多模态复活）

---

## Slide 17: 本节小结

```
┌──────────────────────────────────────────────┐
│ He et al. 2022 - 统一视角                     │
│                                              │
│ 核心定理:                                     │
│   所有 PEFT = h + f(W_down · x) · W_up       │
│                                              │
│ 三大方法等价证明:                             │
│   Prefix Tuning ≡ Parallel Adapter           │
│   LoRA ≡ 去非线性 Adapter                    │
│   (IA)³ ≡ 对角化 Adapter                     │
│                                              │
│ MAM = Prefix(attn) + Parallel(FFN)           │
│   实验上最优 (+1.2 ROUGE-L on XSum)          │
│                                              │
│ ↓                                            │
│ 本专题理论高峰，恭喜你！                       │
└──────────────────────────────────────────────┘
              ↓
       下节: K-Adapter + MAD-X (应用层)
```
