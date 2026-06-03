# Lecture 9: 三线综合 — Prompt + LoRA + Adapter 统一视角

> 论文（基础）: He et al. 2022, "Towards a Unified View of Parameter-Efficient Transfer Learning" (ICLR)
> 论文（补充）: Mao et al. 2022, "UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning" (ACL)
> 配套 notebook: `notebooks/09-three-line-unification.ipynb`

**本节是 PEFT 三大主线学习的高潮。**

---

## Slide 1: 本节路线

```
回顾三大主线 → MAM 统一公式深化 → 数学等价证明 → 28 方法横向对比 → 工程决策树
```

**学完本节，你应该能**:
1. 用一个公式描述 28 种 PEFT 方法
2. 严格证明 Prefix Tuning ≡ Parallel Adapter
3. 在任何场景下做出正确的 PEFT 工程选择

---

## Slide 2: 你已经走完的路

```
┌─────────────────────────────────────────────────┐
│  Prompt-tuning-family  (输入端 PEFT)              │
│   01 Prompt Tuning                              │
│   02 Prefix Tuning                              │
│   03 P-Tuning v1                                │
│   04 P-Tuning v2                                │
│   05 跨方法对比                                  │
├─────────────────────────────────────────────────┤
│  LoRA-family  (权重端 PEFT)                       │
│   01 LoRA + rsLoRA + LoRA+                      │
│   02 AdaLoRA                                    │
│   03 PiSSA + OLoRA                              │
│   04 VeRA                                       │
│   05 LoHa + LoKr                                │
│   06 QLoRA                                      │
│   07 LoftQ                                      │
│   08 DoRA                                       │
├─────────────────────────────────────────────────┤
│  Adapter-tuning-family  (结构端 PEFT)             │
│   01 Houlsby + Pfeiffer                         │
│   02 AdapterFusion                              │
│   03 AdapterDrop + Compacter                    │
│   04 Parallel Adapter                           │
│   05 (IA)³                                      │
│   06 MAM Adapter                                │
│   07 K-Adapter + MAD-X                          │
│   08 AdaMix                                     │
└─────────────────────────────────────────────────┘

= 28 种 PEFT 方法
```

---

## Slide 3: 统一公式（核心）

He et al. (MAM Adapter 论文) 证明:

**所有 PEFT 方法都可以写成**：

$$h \leftarrow h + \Delta h$$

其中 $\Delta h$ 是一个低秩扰动:

$$\Delta h = f(\mathbf{W}_{\text{down}} \cdot x) \cdot \mathbf{W}_{\text{up}}$$

差异只在 4 个轴上:
1. **$f$**: 是 identity / σ / softmax / attention 还是别的
2. **$\mathbf{W}_{\text{down}}, \mathbf{W}_{\text{up}}$**: 矩阵 / 对角 / 共享 / PHM 等
3. **位置**: attention K/V / attention output / FFN intermediate / FFN output
4. **组合方式**: 并联 / 串联 / 残差缩放

---

## Slide 4: 三线在统一公式下的归约

### Prompt-based 主线
- **Prompt Tuning**: 只在 input embedding 加 soft tokens（隐含 $\Delta h_{\text{emb}}$）
- **Prefix Tuning**: 在 K, V 前拼接 prefix vectors → 等价于 attention 的 $\Delta h$
- **P-Tuning v2**: 每层都加 prefix → 多层 $\Delta h_{\text{attn}}^{(l)}$

### Weight-based 主线 (LoRA)
- **LoRA**: $\Delta W = \alpha/r \cdot W_{up} W_{down}$，$f$ = identity
- **DoRA**: 同 LoRA 但加 magnitude × direction 分解
- **VeRA**: 共享 $W_{up}, W_{down}$，每层只学对角缩放
- **(IA)³**: $W_{up} = W_{down} = I$（对角阵），只学缩放向量

### Structure-based 主线 (Adapter)
- **Houlsby**: $\Delta h = W_{up} \sigma(W_{down} h)$，串联
- **Parallel**: 同 Houlsby，**并联**
- **Compacter**: $W_{down}, W_{up}$ 用 PHM 表示
- **MAM**: Prefix (attn) + Parallel (FFN) 组合

---

## Slide 5: 严格证明 — Prefix Tuning ≡ Parallel Adapter

**Prefix Tuning** 在 attention 上的效果:

$$\text{attn}(Q, [P_k; K], [P_v; V])$$

展开 softmax:

$$= \text{softmax}\left(\frac{Q [P_k; K]^T}{\sqrt{d}}\right) [P_v; V]$$

分子可拆为两块:

$$= \frac{[\exp(Q P_k^T); \exp(Q K^T)]}{\sum \exp(\cdot)} [P_v; V]$$

定义 $\lambda(Q) = \frac{\sum \exp(Q P_k^T)}{\sum \exp(Q P_k^T) + \sum \exp(Q K^T)}$，可得:

$$= (1 - \lambda) \cdot \underbrace{\text{attn}(Q, K, V)}_{\text{原 attention}} + \lambda \cdot \underbrace{\text{attn}(Q, P_k, P_v)}_{\text{prefix attention} = \Delta h_{\text{attn}}}$$

→ **Prefix Tuning 就是在 attention 输出上做加性扰动 = Parallel Adapter 形式**。

---

## Slide 6: 严格证明 — LoRA ≡ Parallel Adapter (无非线性)

**Parallel Adapter** 形式:
$$\Delta h = W_{up} \cdot \sigma(W_{down} \cdot x)$$

**当 $\sigma = \text{identity}$**:
$$\Delta h = W_{up} W_{down} x = (W_{up} W_{down}) \cdot x = \Delta W \cdot x$$

其中 $\Delta W = W_{up} W_{down}$，rank $\le \min(r, d)$。

**这正是 LoRA 的公式** (省略缩放因子 $\alpha/r$)。

→ **LoRA = 去掉激活函数的 Parallel Adapter**。

---

## Slide 7: (IA)³ ≡ Adapter with Diagonal Matrices

**(IA)³**:
$$K \leftarrow K \odot \ell_k = K \cdot \text{diag}(\ell_k)$$

写成统一公式形式:
- $W_{down} = \text{diag}(\ell_k - 1) / \alpha$
- $W_{up} = \alpha \cdot I$
- $f = \text{identity}$
- 扰动 $\Delta K = K \cdot \text{diag}(\ell_k - 1) = K \odot \ell_k - K$

→ (IA)³ 是 **对角矩阵化** 的 LoRA。

---

## Slide 8: VeRA ≡ (IA)³ + 共享投影

**VeRA**:
$$\Delta W = \Lambda_d \odot B \Lambda_b A$$

其中 $A, B$ 全局共享（所有层共用），只学对角 $\Lambda_d, \Lambda_b$。

与 (IA)³ 对比:
- (IA)³: 直接学 K, V, FFN 的对角缩放
- VeRA: 通过共享 $A, B$ 投影后再学对角

→ VeRA 是 **加了共享 projection 的 (IA)³**。

---

## Slide 9: 28 方法横向对比表

| # | 主线 | 方法 | $W_{\text{down}}$ | $f$ | $W_{\text{up}}$ | 位置 | 参数 (GPT-2) | 可合并 |
|---|------|------|-------|----|----|------|---------|--------|
| 1 | Prompt | Prompt Tuning | (输入端 embedding) | - | - | input | 7.7K | ❌ |
| 2 | Prompt | Prefix Tuning | $(l, d)$ | attn | (l, d) | K,V prepend | 553K | ❌ |
| 3 | Prompt | P-Tuning v1 | LSTM/MLP | - | - | input | 1M+ | ❌ |
| 4 | Prompt | P-Tuning v2 | per-layer prefix | attn | per-layer | each layer | 184K | ❌ |
| 5 | LoRA | **LoRA** | $(r, d)$ | id | $(d, r)$ | parallel | 295K | ✅ |
| 6 | LoRA | rsLoRA | 同 LoRA | id | 同 LoRA | parallel | 295K | ✅ |
| 7 | LoRA | LoRA+ | 同 LoRA | id | 同 LoRA + 不同 lr | parallel | 295K | ✅ |
| 8 | LoRA | AdaLoRA | $P\Lambda$ | id | $Q^T$ | parallel | 442K | ✅ |
| 9 | LoRA | PiSSA | SVD init | id | SVD init | parallel | 295K | ✅ |
| 10 | LoRA | OLoRA | QR init | id | QR init | parallel | 295K | ✅ |
| 11 | LoRA | **VeRA** | 共享 A + diag | id | 共享 B + diag | parallel | 31K | ✅ |
| 12 | LoRA | LoHa | $(B_1 A_1) \odot (B_2 A_2)$ | id | (combined) | parallel | 590K | ✅ |
| 13 | LoRA | LoKr | $B \otimes A$ | id | (combined) | parallel | 24K | ✅ |
| 14 | LoRA | **QLoRA** | NF4(W) + (r,d) | id | (d,r) | parallel | 295K | ✅ |
| 15 | LoRA | LoftQ | NF4 + SVD | id | SVD | parallel | 295K | ✅ |
| 16 | LoRA | **DoRA** | (r,d) + magnitude | id+norm | (d,r) | parallel | 304K | ✅ |
| 17 | Adapter | Houlsby | (r,d) | σ | (d,r) | series×2 | 609K | ❌ |
| 18 | Adapter | Pfeiffer | (r,d) | σ | (d,r) | series×1 | 304K | ❌ |
| 19 | Adapter | AdapterFusion | (d,d) Q/K/V | softmax | combined | series | 21M | ❌ |
| 20 | Adapter | AdapterDrop | (r,d) | σ | (d,r) | drop k | 304K | ❌ |
| 21 | Adapter | **Compacter** | PHM | σ | PHM | series | 83K | ❌ |
| 22 | Adapter | Parallel | (r,d) | σ | (d,r) | parallel | 304K | ❌ |
| 23 | Adapter | **(IA)³** | diag | id | id | element-wise | 55K | ✅ |
| 24 | Adapter | **MAM** | mixed | mixed | mixed | mixed | 857K | ❌ |
| 25 | Adapter | K-Adapter | (r,d)×N | σ | (d,r) | sum | 609K | ❌ |
| 26 | Adapter | MAD-X | LA+TA stack | σ | (d,r) | stack | 1.22M | ❌ |
| 27 | Adapter | AdaMix | (r,d)×N | σ | (d,r) | random/avg | 1.22M | ❌ |
| 28 | Adapter | UniPELT | gated mix | mixed | mixed | parallel | 880K | partial |

---

## Slide 10: PEFT 工程决策树（终极版）

```
你的场景是什么？
│
├─ 65B 模型 + 单卡 24GB GPU
│  → QLoRA ⭐⭐⭐
│
├─ 追求最高质量
│  → DoRA (单方法) / MAM (混搭) ⭐⭐⭐
│
├─ 极致省参 (<0.05%)
│  → (IA)³ / VeRA ⭐⭐⭐
│
├─ 多任务千用户
│  → VeRA (LoRA 系) / MAD-X (Adapter 系) ⭐⭐
│
├─ Stable Diffusion 风格微调
│  → LoKr (Kronecker 分解) ⭐⭐⭐
│
├─ Few-shot 学习
│  → (IA)³ / Prompt Tuning ⭐⭐
│
├─ NER / 序列标注
│  → P-Tuning v2 (独家) ⭐⭐⭐
│
├─ 多模态（vision + language）
│  → LLaMA-Adapter / Q-Former (下节展开) ⭐⭐⭐
│
├─ 推理时延敏感（必须 0 时延）
│  → LoRA 系（可合并）⭐⭐
│
├─ 训练慢
│  → PiSSA / DoRA (2× 加速) ⭐⭐
│
├─ 大 r 不稳
│  → rsLoRA (α/√r) ⭐
│
├─ 多任务组合
│  → AdapterFusion / AdaMix ⭐⭐
│
├─ 多语言迁移
│  → MAD-X ⭐⭐⭐
│
└─ 实验对比 baseline
   → LoRA r=8（社区标准）⭐
```

---

## Slide 11: 跨主线"等价对"

| 主线 A 方法 | ≡ 主线 B 方法 | 等价条件 |
|------------|--------------|---------|
| Prefix Tuning (Prompt) | Parallel Adapter (Adapter) | 注入位置 = attention |
| LoRA (Weight) | Parallel Adapter (Adapter) | σ = identity |
| (IA)³ (Adapter) | LoRA r=1 + diag | 退化形式 |
| VeRA (LoRA) | (IA)³ + 共享投影 | 增强形式 |
| DoRA (LoRA) | LoRA + magnitude norm | 增强形式 |

`★ Insight`：**Adapter 是 PEFT 的"祖先"**。LoRA 实际上是 Adapter 的子集（去非线性），Prompt 的 prefix 在 attention 上等价于 Adapter 注入。理解 Adapter 等于理解一半 PEFT。

---

## Slide 12: UniPELT — 自适应混搭

Mao et al. 2022 提出 UniPELT，用门控网络自动选择何时用 LoRA/Prefix/Adapter:

$$h = \alpha_{\text{LoRA}}(x) \cdot h_{\text{LoRA}} + \alpha_{\text{Prefix}}(x) \cdot h_{\text{Prefix}} + \alpha_{\text{Adapter}}(x) \cdot h_{\text{Adapter}}$$

其中 $\alpha$ 是 input-dependent gates。

→ "学会"用哪种 PEFT。比 MAM 更动态，但训练复杂度高。

---

## Slide 13: 思考题

**公式题**:
1. 推导：在什么条件下，DoRA 退化为 LoRA？
2. 证明：(IA)³ 的对角阵限制了"扰动 rank"，最大 rank 多少？
3. UniPELT 的 gating $\alpha$ 怎么训？为什么不会全选 LoRA 或全选 Adapter？

**设计题**:
4. 设计一个新方法：把 DoRA 和 (IA)³ 组合。它叫什么？参数多少？
5. 在 LLaMA-70B 上用 QLoRA + DoRA + (IA)³ 三层组合，预期收益是？
6. 假设你要发新 PEFT 论文，给定 28 方法，下一个突破点在哪？

**对比题**:
7. 用一句话解释为什么"统一视角"重要——给一个还没学 PEFT 的同学。
8. PEFT 三大主线在 2025+ 哪个会持续活跃？为什么？

---

## Slide 14: 与下节衔接 — L10 PEFT 下一步

**到此为止**: 完成 28 方法的统一理解。

**下节**: PEFT 的未来在哪？
1. Adapter 的多模态复活（LLaMA-Adapter, Q-Former）
2. 长上下文专题（LongLoRA, YaRN）
3. 对齐专题（DPO, SimPO）
4. MoE 主线（DeepSeek, Mixtral）

为整个 PEFT 学习画上句号，并指向下一程。

---

## Slide 15: 本节小结

```
┌──────────────────────────────────────────────────┐
│ 三线综合 — PEFT 的统一视角                          │
│                                                  │
│ 核心公式: h ← h + f(W_down · x) · W_up           │
│                                                  │
│ 三大主线在数学上是同一种东西的不同伪装:            │
│   Prompt = attention 端注入                       │
│   LoRA = 去非线性 Adapter                         │
│   Adapter = 带非线性 LoRA + 不同位置              │
│                                                  │
│ 工程上仍要分场景选用 (28 方法决策树)              │
│                                                  │
│ ↓                                                │
│ 主线学习完结，恭喜你掌握了 PEFT 全景！             │
└──────────────────────────────────────────────────┘
              ↓
       下节: L10 PEFT 下一步 (多模态复活 + 路线图)
```
