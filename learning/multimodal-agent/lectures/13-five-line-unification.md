# L13 · 五线综合统一 ⭐⭐⭐⭐⭐ 系列理论高峰

> 32 slides | 90 min | Multimodal Agent 第 13 讲 — **整个 PEFT + RL 学习系列的理论高峰**

---

## Part I · 五线回顾（8 slides, 20 min）

---

## Slide 1 · 五线总览

| # | 线 | 专题 | 核心 idea |
|---|---|------|---------|
| 1 | Prompt | PEFT 系列 1 | input embedding 扰动 |
| 2 | LoRA | PEFT 系列 2 | weight 低秩扰动 |
| 3 | Adapter | PEFT 系列 3 | structure 加层扰动 |
| 4 | RLHF | RL 系列 2 | distribution shape 改 |
| 5 | R1 | RL 系列 5 | trajectory exploration 改 |

→ 五条线 = 改大模型行为的五个层面。

---

## Slide 2 · Prompt 线（专题 1）

- **典型方法**：Prefix Tuning / P-Tuning v2 / Soft Prompt
- **改的位置**：input embedding 序列前缀
- **可训参数**：~0.01%
- **几何**：input 空间局部扰动

`y = LM(prefix + x)`

→ 最轻量，但能力上限低。

---

## Slide 3 · LoRA 线（专题 2）

- **典型方法**：LoRA / QLoRA / VeRA / DoRA / NoLA / Pissa
- **改的位置**：每层 attention 的 Q/K/V/O 权重
- **可训参数**：~0.1-1%
- **几何**：weight 矩阵的低秩子空间

`W' = W + (BA), rank(BA) = r << min(d_in, d_out)`

→ 性价比最高，工业最常用。

---

## Slide 4 · Adapter 线（专题 3）

- **典型方法**：Houlsby / Pfeiffer / Compacter / AdaMix / IA³
- **改的位置**：每层 FF 后插入新模块
- **可训参数**：~0.5-3%
- **几何**：feature 空间的非线性变换

`h' = h + Adapter(h)`

→ 灵活但参数稍多。

---

## Slide 5 · RLHF 线（专题 2 of RL）

- **典型方法**：InstructGPT / DPO / KTO / ORPO
- **改的方式**：用 RM/preference 改变 LLM 的输出分布
- **几何**：把 distribution 推向 chosen，拉离 rejected

`max E[r(y|x)] - β · KL(π || π_ref)`

→ 让 LLM "学会人想要的回答"。

---

## Slide 6 · R1 线（专题 5 of RL）

- **典型方法**：GRPO / DAPO / VAPO / PRIME
- **改的方式**：用 verifier reward 改变 trajectory 探索
- **几何**：在 token-level state space 上，把 "好 trajectory" 概率推高

`max E_{rollout}[r_format + r_accuracy] s.t. KL(π || π_ref) small`

→ 让 LLM "学会推理"（aha moment）。

---

## Slide 7 · 五线在 stack 中的位置

```
                  [生成 token y]
                       ↑
       [LM forward: 概率分布 p(y|x; θ_LM, φ)]
                       ↑
       [3 个 PEFT 改 θ_LM 不同位置]
       [2 个 RL 改 sample 分布 / trajectory]
```

→ φ 是个 abstract 参数，五线只是 φ 的不同安放。

---

## Slide 8 · 三句话一锤定音

> **PEFT 改 model.**（Prompt / LoRA / Adapter 改的是 θ_LM 与 φ）
> **RLHF 改 distribution.**（用 RM 把分布推向 chosen）
> **R1 改 trajectory.**（用 verifier 让 LLM 在 token level 探索新路径）

→ 五条线 = 三类干预。

---

## Part II · 统一公式（12 slides, 35 min）

---

## Slide 9 · 核心命题

任意 LLM 适配方法可以写成：

```
y ~ p(y | x ; θ_LM , φ)
```

- `θ_LM`：预训练 LLM 参数（通常 freeze）
- `φ`：适配参数（五线只是 φ 的不同放法）

→ 这是统一视角。

---

## Slide 10 · Prompt 是 input 扰动

φ = `[p_1, p_2, ..., p_k]`（k 个 soft prompt embedding）

```
y ~ LM([p_1, ..., p_k, e(x_1), e(x_2), ...])
```

φ 在最底层（embedding 之前）作用。

---

## Slide 11 · LoRA 是 weight 扰动

φ = `{B_i, A_i for each layer i}`（每层一对低秩矩阵）

```
W_i' = W_i + B_i A_i              # 每层 attention 权重
```

φ 在中层（每个 transformer block）作用。

---

## Slide 12 · Adapter 是 structure 扰动

φ = `{Adapter_i for each layer i}`（每层 down/up 投影）

```
h_i' = h_i + Adapter_i(h_i)       # 每层 feature
```

φ 是结构增量，相当于"加层"。

---

## Slide 13 · RLHF 是 distribution 改

φ = RM 的训练 + KL ref penalty 的调谐

```
π_RLHF(y|x) ∝ π_ref(y|x) · exp(r_RM(x, y) / β)
```

φ 在采样分布层面起作用 —— 不直接改 LLM 参数，而是改它输出的"分布形状"。

---

## Slide 14 · R1 是 trajectory 探索

φ = verifier + group rollout 机制

```
π_R1(y|x) = argmax E_rollouts [r_format(y) + r_accuracy(y)]
```

φ 改的是 LLM 在 trajectory 空间的探索倾向 —— 多次 rollout 后留下"成功的 trajectory pattern"。

---

## Slide 15 · 五线对比矩阵

| 线 | φ 的形态 | 影响层次 | 训完 LLM 参数变化 |
|---|---------|---------|-----------------|
| Prompt | k 个 embedding | input | ✗ |
| LoRA | 每层 BA 对 | weight | ✗ (实际 + ΔW) |
| Adapter | 每层 module | structure | ✗ (实际 + Adapter) |
| RLHF | RM + 训练数据 | distribution | ✓ 全量 fine-tune |
| R1 | verifier + rollout | trajectory | ✓ 全量 / LoRA |

---

## Slide 16 · 等价对：Prefix ≡ Parallel Adapter

Prefix Tuning 与 Parallel Adapter 在数学上等价（专题 3 已证）：

```
Prefix: K = [P_K ; X·W_K], V = [P_V ; X·W_V]
Parallel Adapter: h += Linear(X · W_K, ...)
```

→ 表面 input vs structure，深层等价。

---

## Slide 17 · 等价对：LoRA(σ=id) ≡ Parallel Adapter

LoRA 不加激活时：

```
LoRA: W += BA  → h += X·BA
Parallel Adapter (no σ): h += A·B(h)
```

shapes 和顺序略不同但本质一样。

→ "weight 扰动" 与 "结构扰动" 在 linear 极限下相同。

---

## Slide 18 · 等价对：DPO ≡ 隐式 RM 的 BT

DPO 的 loss：

```
r(x, y) = β · log(π(y|x) / π_ref(y|x))
L_DPO = -log sigmoid(r_chosen - r_rejected) = L_BT
```

→ DPO 是把 RM 的 reward 用 policy ratio 表示后做 BT loss。隐式 RM。

---

## Slide 19 · 等价对：GRPO ≡ PPO 去 critic

GRPO 用 group baseline 代替 critic V：

```
PPO:  A = G_t - V(s_t)
GRPO: A_i = (R_i - R̄) / σ  (response-level)
```

→ "去 critic" 是 PPO 的简化变体，本质相同。

---

## Slide 20 · 大一统公式

```
y ~ p(y | x ; θ_LM , φ)
        ↓
   φ 在不同位置安放：
     - input → Prompt
     - weight → LoRA
     - structure → Adapter
     - distribution → RLHF
     - trajectory → R1
```

**一行公式**：所有 LLM 适配 = 找一个 φ，让 LLM 输出更接近"期望"。

---

## Part III · 工程选型决策树（8 slides, 20 min）

---

## Slide 21 · 决策树（高层）

```
任务类型：
  ├── 通用 chat / 客服      → RLHF 或 DPO
  ├── 推理 / 数学 / 代码    → R1 (GRPO / DAPO)
  ├── 风格 / 个性化         → LoRA 或 Prompt
  ├── 多任务 / 模块化       → Adapter
  ├── 跨语言 / Few-shot     → Prompt
  └── 单一窄领域            → LoRA + 1 epoch SFT
```

---

## Slide 22 · 场景 1 · 客服 chatbot

- **目标**：refuse 危险问题 + 简洁回答 + 客气
- **数据**：偏好对 + harmless 标注
- **推荐**：DPO（专题 3）
- **理由**：开放任务 + 工程简单 + 与 RLHF 效果接近

---

## Slide 23 · 场景 2 · 数学竞赛 / AIME

- **目标**：长链推理 + 高准确
- **数据**：数学题 + ground_truth
- **推荐**：R1-Zero (GRPO + rule reward) + DAPO 4 件套
- **理由**：rule-based reward 完美适用，aha moment 涌现

---

## Slide 24 · 场景 3 · SWE Agent

- **目标**：读 issue + 生成 patch + 通过测试
- **数据**：SWE-Gym (issue + fix)
- **推荐**：R1-style + SWE-Gym training（专题 7 L05）
- **理由**：通过测试是 verifier reward，与 R1 同构

---

## Slide 25 · 场景 4 · 千用户 SaaS

- **目标**：每个用户独立微调 + 显存共享 base
- **数据**：每用户少量
- **推荐**：LoRA + multi-LoRA serving（vllm 支持）
- **理由**：每用户只需存 1MB LoRA weights

---

## Slide 26 · 选型矩阵

| 场景 | 选 | 显存 | 训练时长 |
|------|---|------|---------|
| 客服 | DPO | 2× | 一天 |
| 数学 | R1-GRPO | 3× | 一周 |
| SWE | R1-GRPO + verifier | 3× | 一周 |
| SaaS | LoRA | 1× | 一小时 |
| 跨语言 | Prompt | < 1× | 几小时 |
| 多任务 | Adapter (AdapterFusion) | 1× | 一天 |

---

## Slide 27 · 何时叠加多条线

工业 LLM 经常 **叠加**：
- LoRA + DPO：用 LoRA 训 actor，省显存
- LoRA + R1-GRPO：Qwen-1.5B 挑战轨 capstone 用法
- Adapter + RLHF：多任务 + 对齐

→ "改 model" + "改 distribution / trajectory" 可同时进行。

---

## Slide 28 · 五线 + 资源约束（决策树扩展）

```
显存 < 8 GB → LoRA only / Prompt only
显存 8-24GB → LoRA + DPO / LoRA + GRPO
显存 24-80GB → Full + DPO / Full + GRPO
显存 > 100GB → Full + RLHF (PPO) / R1
```

---

## Part IV · 历史观 + 下一程（4 slides, 15 min）

---

## Slide 29 · 大模型对齐 5 年史

```
2018  GPT-1 transformer
2019  GPT-2 prompt engineering
2020  GPT-3 few-shot
2021  T0 / FLAN  (instruction tuning 起源)
2022  Prefix Tuning / LoRA / Adapter (PEFT 三轴)
2022  InstructGPT (RLHF 范式)
2023  GPT-4 / DPO / RLAIF / Anthropic CAI
2024  GRPO / Self-Reward / o1 / Math-Shepherd / TinyZero 预备
2025  R1 / R1-Zero / Kimi k1.5 / DAPO / VAPO / VLM-R1 / Open-R1 / Spurious Rewards
2026  当前学习节点：你正在毕业 ✓
```

---

## Slide 30 · 下一程预测

接下来的 1-2 年（2026-2028）：

1. **MoE (Mixture of Experts)**：Mixtral / DeepSeek-MoE 的扩散
2. **长上下文**：1M / 10M token，对推理 RL 影响巨大
3. **Continuous Pretraining**：domain adaptation 与 RL 结合
4. **World Model RL**：在 simulated env 中 long-horizon RL
5. **Agent RL 工业化**：Computer-use / Browser-use / Coding agent

→ 五线 + R1 + 多模态 + Agent 仍是基础。

---

## Slide 31 · 系列毕业 checklist

- [ ] PEFT 三专题（Prompt / LoRA / Adapter）✓
- [ ] RL 基础（PPO）✓
- [ ] RLHF Classic（InstructGPT 三段管线）✓
- [ ] DPO 家族（DPO/IPO/KTO/ORPO/SimPO/...）✓
- [ ] Process Reward（PRM/RLVR）✓
- [ ] R1 时代（GRPO/R1-Zero/Kimi/TinyZero）⭐⭐⭐⭐⭐ ✓
- [ ] RL SOTA 2026（DAPO/VAPO/PRIME/Skywork V2）✓
- [ ] Multimodal Agent（VLM-R1/WebRL/SWE-Gym/...）✓
- [ ] **五线综合理论统一**（本讲）✓
- [ ] 毕业作品（L14 五个 ckpt 同题对照）→ 下一讲

🎓 **128 方法 / 90 lecture / ~130h 学完。**

---

## Slide 32 · 一句话总结

> 大模型适配的本质：**在不动 θ_LM 太多的情况下，找一个 φ，把模型行为推到期望方向**。
>
> 五线只是 φ 的不同安放位置。
>
> 你已掌握全部五条。

---

🎓 **下一讲 L14 毕业 Capstone — 同一道 GSM8K 题用 5 个 ckpt 对照。**
