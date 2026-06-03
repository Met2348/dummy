# Lecture 10: PEFT 的下一步 — Adapter 多模态复活 + 后续路线图

> 配套 notebook: `notebooks/10-peft-next-step.ipynb`

**本节是整个 PEFT 学习的终章。**

---

## Slide 1: 本节路线

```
回顾 → Adapter 死亡（2022-）→ 多模态复活（LLaMA-Adapter / Q-Former / LLaVA）→ 路线图
```

**学完本节，你应该能**:
1. 解释 2022 后 Adapter 创新枯竭的 5 原因
2. 理解 Adapter 在多模态时代如何"重生"
3. 规划下一步学习的优先级（长上下文 / 对齐 / MoE / 推理）

---

## Slide 2: 现状 — 2022 后 Adapter 范式断更

你应该已经发现：本系列 11 种方法**最后一个是 2022 年的 AdaMix**。

为什么 2023-2026 没有新的 Adapter 方法？

---

## Slide 3: 五个原因 — Adapter 范式为什么"死"了

### 原因 1: LoRA 范式吞噬 (paradigm subsumption)
- LoRA = 去掉非线性的 Adapter，但可合并 → 推理零延迟
- 这一个优势在 LLM 时代决定性
- 2023+ "PEFT 创新" 99% 都是 LoRA 衍生

### 原因 2: 结构性 limitation
- 串联非线性必然增加每层 forward 时延
- LLM 推理时代（context length 是金）：致命缺陷

### 原因 3: (IA)³ 已是参数效率天花板
- 3 个对角缩放向量已经到极限
- 再压只剩 BitFit（只调 bias，无创新空间）

### 原因 4: 行业重心整体迁移
| 时期 | 研究热点 |
|------|---------|
| 2019-2021 | Adapter 探索 |
| 2021-2022 | LoRA 兴起 |
| 2023 | 量化 (QLoRA, LoftQ, GPTQ, AWQ) |
| 2024 | LoRA 衍生 (DoRA, PiSSA) + 长上下文 (LongLoRA) |
| 2025-2026 | 对齐 (DPO, SimPO) + 推理优化 (vLLM) |

### 原因 5: 工业落地默认 LoRA
- HuggingFace peft 主推 LoRA 系
- AdapterHub 的 adapters 库维护节奏放缓
- 社区 inertia 形成

---

## Slide 4: 但 Adapter 没死 — 它"变形复活"了

`★ Insight`：**Adapter 思想在多模态领域大放异彩**。

在多模态（Vision + Language）领域:
- "跨模态对齐"天然需要"插入新模块" → Adapter 强项
- LoRA 的"低秩 + 同模态"假设在跨模态不成立
- Adapter 的"加层 + 桥接"思想完美匹配

→ 2023+ 的"伪 Adapter"工作大爆发。

---

## Slide 5: LLaMA-Adapter (2023)

**论文**: Zhang et al. 2023, "LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention"

**核心 idea**:
1. 在 LLaMA 每层加 learnable prefix tokens（类 Prefix Tuning）
2. 用 **zero-init gating** 控制 prefix 影响（初始 = 0，逐渐学习）
3. 训练时只学 prefix + gate → 极致 PEFT

**结构**:
```
x → LLaMA layer
   ↓
attention(x, prefix_k, prefix_v) × gate   ← 零初始化 gate
   ↓
+ x
```

**参数量**: 1.2M (LLaMA-7B 的 0.02%)

---

## Slide 6: LLaMA-Adapter V2 (2023)

**升级**:
1. **多模态扩展**: 加 vision encoder（ViT/CLIP）+ 视觉 token → LLM
2. **解锁 bias tuning**: 部分冻结+部分调整
3. **指令数据**: 在 Stanford Alpaca 上训练

**用法**:
```
图片 → CLIP → vision_tokens
                ↓
[vision_tokens; instruction tokens] → LLaMA + LLaMA-Adapter
                ↓
            response
```

→ "视觉 token = adapter 的特殊形式"。

---

## Slide 7: Q-Former (BLIP-2, 2023)

**论文**: Li et al. 2023, "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs"

**Q-Former (Querying Transformer)** 架构:
```
Image (frozen ViT) → image_tokens (~257 个)
                                   ↓
              ┌────────────────────┘
              ↓
Q-Former: 32 learnable query tokens + cross-attention
              ↓
              └─→ 32 visual tokens
                       ↓
                  Frozen LLM
                       ↓
                  generated text
```

**这就是 adapter 的现代形式**:
- 32 个 query tokens = learnable "bottleneck"
- cross-attention = "把图片信息压缩到 32 token"
- LLM 完全冻结 → 极致 PEFT

---

## Slide 8: LLaVA Projector (2023)

**论文**: Liu et al. 2023, "Visual Instruction Tuning"

**LLaVA** = 最简 adapter:
```
Image → CLIP-ViT → image_features (576 tokens × 1024)
                                ↓
                  ┌─────────────┘
                  ↓
            单层 MLP projector
                  ↓
            image_tokens (576 × 4096)  ← 投影到 LLaMA dim
                  ↓
       [image_tokens; instruction] → Frozen LLaMA
                  ↓
                response
```

**Projector 参数**: 1024 × 4096 + 4096 = 4.2M

→ **这就是一个 adapter** — 在视觉-语言桥接位置插入。

---

## Slide 9: AdapterSoup (2023)

**论文**: Chronopoulou et al. 2023, "AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models"

**核心 idea**: 多个 domain adapter 权重平均（类 Model Soup）

```
adapter_1 (medical) ┐
adapter_2 (legal)   ├─→ average weights → adapter_general
adapter_3 (finance) ┘
```

**优势**:
- 不需要训练 fusion 层（vs AdapterFusion）
- 推理零开销（已 merge）
- 性能接近 ensemble

→ "AdaMix 的多领域版"。

---

## Slide 10: MiniGPT-4 / InstructBLIP / 现代多模态 (2023-2024)

整个 multimodal LLM 范式:

| 模型 | adapter 部分 |
|------|------------|
| **MiniGPT-4** | Q-Former (BLIP-2) + 单层 linear |
| **InstructBLIP** | Q-Former + instruction tuning |
| **LLaVA-1.5** | 双层 MLP projector |
| **Idefics-2** | Perceiver Resampler (类 Q-Former) |
| **Qwen-VL** | Cross-attention adapter |
| **GPT-4V** (闭源推测) | 类 Q-Former |

**共性**: 都用 "adapter-like" 模块连接视觉 encoder 和 LLM。

→ **2023-2024 多模态时代，Adapter 思想是事实标准**。

---

## Slide 11: 后续专题路线图

### 优先级 1: **长上下文** (LongLoRA / PI / YaRN)
- 训练 LLM 处理 128K+ context
- LongLoRA = LoRA + sparse attention (S^2-Attn)
- PI (Position Interpolation), YaRN, NTK 等位置编码扩展

### 优先级 2: **对齐** (RLHF / DPO / SimPO)
- 让模型符合人类偏好
- RLHF: PPO + reward model
- DPO: 直接偏好优化（无 reward model）
- SimPO, ORPO, KTO 等改进

### 优先级 3: **MoE** (Mixtral / DeepSeek-MoE / OLMoE)
- Mixture of Experts
- Top-k routing
- Expert parallelism

### 优先级 4: **推理优化** (vLLM / FlashAttention / Continuous Batching)
- 部署级优化
- KV cache 管理
- PagedAttention

---

## Slide 12: 你的下一步推荐

如果你要选下一个专题，按"基于书本主线"推荐:

```
书本主线（《大模型算法》）章节顺序:
  Ch1  Background           ✅ (基础知识，跳过)
  Ch2  PEFT                  ✅ (3 专题已完成: Prompt + LoRA + Adapter)
  Ch3  Alignment             ⭐⭐⭐ RECOMMENDED NEXT
       3.1 RLHF
       3.2 DPO
       3.3 SimPO/ORPO/KTO
  Ch4  Long Context          ⭐⭐
  Ch5  Reasoning             ⭐⭐
  Ch6  Inference Opt         ⭐
```

`★ 建议`：**优先做对齐专题** (Ch3)，因为：
1. 接续书本主线
2. 现代 LLM 必须经过对齐
3. RLHF → DPO 是这两年最热的技术演化

---

## Slide 13: 整个 PEFT 学习的 takeaway

经过三个专题，你应该带走:

```
1. 理论
   - PEFT 三大主线（输入/权重/结构）
   - 统一公式 (h ← h + f(W_down · x) · W_up)
   - 主线之间的等价证明

2. 工程
   - 28 方法横向对比
   - 工程选型决策树
   - 推理时延 vs 参数量 trade-off

3. 代码
   - minimal 手写 (30+ 个 .py)
   - peft / adapters 库调包对照
   - 一致性测试 (logits 误差 ≤ 1e-7)

4. 历史
   - Adapter (2019) → LoRA (2021) → 量化 (2023) → 多模态 (2024)
   - PEFT 的演化轨迹与未来方向
```

---

## Slide 14: 关于"PEFT 已死"的判断

一种常见误解: "现在大家都用 LoRA，PEFT 没新东西了"。

**实际情况**:
1. **LoRA 主线仍在进化**: 2024 DoRA, MoRA, PiSSA 都是新方法
2. **多模态时代 Adapter 思想王者归来**: Q-Former, LLaVA Projector 是核心
3. **长上下文 PEFT**: LongLoRA 是新方向
4. **对齐 PEFT**: PEFT + DPO 组合（如 LoRA-DPO）

→ PEFT 仍是大模型工程基础，掌握它你已具备"调教大模型"的核心能力。

---

## Slide 15: 思考题

**反思题**:
1. 三个专题中，哪个方法你印象最深？为什么？
2. 哪个方法的数学推导让你"开窍"了？（LoRA / DoRA / PHM / Prefix=Parallel 等）
3. 如果你要发一篇 PEFT 论文，会选什么方向？

**实践题**:
4. 用 DoRA + (IA)³ 组合在 GLUE 上跑（自定）。
5. 实现 LLaMA-Adapter v1 在 GPT-2 上的版本（用 zero-init gate）。
6. 设计一个 "MAM Adapter + DoRA" 的杂交方法，预估参数。

**展望题**:
7. 2025-2026 PEFT 还有哪些"未探索"方向？（gating / 异构 / 长上下文优化 / etc.）
8. 你认为 PEFT 5 年后还会存在吗？为什么？

---

## Slide 16: 致谢

```
┌──────────────────────────────────────────────┐
│   恭喜你完成 PEFT 三大主线学习！               │
│                                              │
│   累计学时:                                   │
│     Prompt-tuning-family   ~6 hours          │
│     LoRA-family            ~10 hours         │
│     Adapter-tuning-family  ~13 hours         │
│     合计                   ~29 hours         │
│                                              │
│   累计方法: 28 种                             │
│   累计代码: 60+ 个 .py 文件                   │
│   累计 lecture: 23 篇                         │
│   累计 notebook: 23 个 ipynb                  │
│                                              │
│   ↓                                          │
│   下一站: 对齐专题 (RLHF / DPO / SimPO)       │
└──────────────────────────────────────────────┘
```

**END**
