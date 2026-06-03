# Adapter Tuning 家族学习专题

> 配套书籍：《大模型算法：强化学习、微调与对齐》(ISBN 9787121500725)，2.3 章 Adapter 系列
> 设计文档：[`../../docs/superpowers/specs/2026-06-03-adapter-tuning-family-design.md`](../../docs/superpowers/specs/2026-06-03-adapter-tuning-family-design.md)
> 实施计划：[`../../docs/superpowers/plans/2026-06-03-adapter-tuning-family.md`](../../docs/superpowers/plans/2026-06-03-adapter-tuning-family.md)
> 承接专题：[`../lora-family/README.md`](../lora-family/README.md) + [`../prompt-tuning-family/README.md`](../prompt-tuning-family/README.md)

---

## 专题概览

本专题覆盖 **11 种 Adapter 家族方法**，组织为 **10 个 lecture**（8 主线 + 2 capstone）。本专题是 **PEFT 三大主线** (Prompt + LoRA + Adapter) 的最后一站，并承担"三线综合"职责。

| Lecture | 主方法 | 附录 | 核心 idea |
|---------|--------|------|----------|
| [01](lectures/01-houlsby-pfeiffer.md) | Houlsby | Pfeiffer | 串联 Adapter 基础 (down→σ→up→+residual) |
| [02](lectures/02-adapterfusion.md) | AdapterFusion | — | 多任务 attention 融合（两阶段训练）|
| [03](lectures/03-adapterdrop-compacter.md) | AdapterDrop, Compacter | — | 推理加速 + PHM 超复数压缩 |
| [04](lectures/04-parallel-adapter.md) | Parallel Adapter | — | 并联结构（LoRA 前身）|
| [05](lectures/05-ia3.md) | (IA)³ | — | 极致压缩（3 个对角缩放向量）|
| [06](lectures/06-mam-adapter.md) | MAM Adapter | — | ⭐ **统一视角**（Prefix + Parallel）|
| [07](lectures/07-k-adapter-mad-x.md) | K-Adapter, MAD-X | — | 知识注入 + 跨语言 |
| [08](lectures/08-adamix.md) | AdaMix | — | Mixture of Adapters（MoE 路由）|
| [09](lectures/09-three-line-unification.md) | 三线综合 | UniPELT | Prompt + LoRA + Adapter 统一公式 + 28 方法决策树 |
| [10](lectures/10-peft-next-step.md) | PEFT 下一步 | — | Adapter 多模态复活 + 后续专题路线图 |

完整专题 ~13 小时（10 lecture × 40-60 min + notebook 实验）。

## 学习路径

```
基础 (L1-L2)
   Houlsby + Pfeiffer (双串联简化) → AdapterFusion (多任务)

效率/压缩 (L3-L5)
   AdapterDrop + Compacter (PHM) → Parallel Adapter → (IA)^3

理论高峰 (L6)
   ⭐ MAM Adapter — 三线统一

应用层 (L7-L8)
   K-Adapter + MAD-X → AdaMix

跨专题综合 (L9-L10)
   三线综合 → PEFT 下一步 (多模态复活)
```

## 目录结构

```
learning/adapter-tuning-family/
├── README.md                         # 本文件
├── environment/
│   ├── requirements.txt              # adapters 1.3 强制 transformers 4.x
│   └── verify_env.py                 # 三段式自检
├── papers/                           # 11 篇论文占位
├── lectures/                         # 10 篇 PPT-style 中文 md
├── src/
│   ├── common.py
│   ├── houlsby_minimal.py / houlsby_adapters.py
│   ├── pfeiffer_minimal.py / pfeiffer_adapters.py
│   ├── adapterfusion_minimal.py / adapterfusion_adapters.py
│   ├── adapterdrop_minimal.py / adapterdrop_adapters.py
│   ├── compacter_minimal.py / compacter_adapters.py     # PHM 数学难点
│   ├── parallel_minimal.py / parallel_adapters.py
│   ├── ia3_minimal.py / ia3_adapters.py / ia3_peft.py   # ⭐ 三轨实现
│   ├── mam_minimal.py                                    # 仅 minimal
│   ├── k_adapter_minimal.py / k_adapter_adapters.py
│   ├── madx_minimal.py / madx_adapters.py
│   ├── adamix_minimal.py                                 # 仅 minimal
│   └── tests/                                            # 8 个测试文件
└── notebooks/                        # 10 个 ipynb
```

## 环境配置

> **重要警告**：本专题的 `adapters` 库 (AdapterHub) **强制依赖 transformers 4.x**，会从 5.9 自动降级到 4.57。这只影响本专题的 venv，不影响 LoRA/Prompt 专题（它们不依赖 transformers 5.x 新特性）。

```powershell
# 已有 LoRA 专题环境（cu130 nightly torch + transformers 5.9）的话:
pip install adapters
# 会自动降级 transformers 5.9 -> 4.57

# 验证
python learning/adapter-tuning-family/environment/verify_env.py
```

预期输出（截至 2026-06-03）：

```
Part A (基础):           PASS    # torch 2.13.0.dev+cu130, transformers 4.57.6, adapters 1.3, peft 0.19
Part B (GPU):            PASS    # RTX 5090, 25.7 GB, sm_120
Part C (adapters lib):   PASS    # Pfeiffer smoke test
```

## 横向对比表（11 方法）

### 1. 参数量与主战场

| 方法 | 年份 | $\Delta h$ 形式 | 参数 (GPT-2, r=16) | 占 base | 可合并 | 主战场 |
|------|------|---------------|-------------------|---------|--------|--------|
| **Houlsby** | 2019 | $x + W_{up}\sigma(W_{down}x)$ × 2 | 608,640 | 0.49% | ❌ | 经典基础 |
| **Pfeiffer** | 2020 | $x + W_{up}\sigma(W_{down}x)$ × 1 | 304,320 | 0.24% | ❌ | 简化版（默认）|
| **AdapterFusion** | 2021 | $\sum_i \alpha_i(Q) \cdot a_i(x)$ | 21,233,664 | 14.5% | ❌ | 多任务组合 |
| **AdapterDrop** | 2020 | Pfeiffer + 随机丢 | 304,320 | 0.24% | ❌ | 推理加速 |
| **Compacter** | 2021 | PHM down/up + 跨层共享 A | 83,264 | 0.07% | ❌ | 极致压缩 |
| **Parallel Adapter** | 2021 | $\text{base}(x) + s W_{up}\sigma(W_{down}x)$ | 304,320 | 0.24% | ❌ | 串/并联对比 |
| **(IA)³** | 2022 | $K \odot \ell_k$, $V \odot \ell_v$, $h_{ff} \odot \ell_{ff}$ | 55,296 | 0.04% | ✅ | 极致省参 + few-shot |
| **MAM Adapter** | 2022 | Prefix(attn) + Parallel(FFN) | 857,280 | 0.69% | ❌ | ⭐ SOTA + 理论高峰 |
| **K-Adapter** | 2020 | $\sum_t a_t(x)$ (多类知识)  | 608,640 (2 类) | 0.49% | ❌ | 知识注入 |
| **MAD-X** | 2020 | LA(x) → TA(x) (Stack)  | 1,221,888 (3 lang) | 0.97% | ❌ | 跨语言 |
| **AdaMix** | 2022 | Random expert (train), avg (infer) | 1,217,280 (N=4) | 0.97% | merge ✓ | MoE 路由 |

### 2. 训练特性

| 方法 | 学习率 | 训练复杂度 | 推理时延 | adapters 库 | peft 库 |
|------|--------|------------|----------|-------------|---------|
| Houlsby | 1e-4 | 低 | +10% | ✅ `houlsby` | — |
| Pfeiffer | 1e-4 | 低 | +5% | ✅ `pfeiffer` | — |
| AdapterFusion | 5e-5 | 高（两阶段） | +50% | ✅ `Fuse(...)` | — |
| AdapterDrop | 1e-4 | 低 | 0~+5%（可控）| ✅ Skip composition | — |
| Compacter | 5e-4 | 中（PHM 矩阵构造）| +5% | ✅ `compacter` | — |
| Parallel Adapter | 1e-4 | 低 | +5% | ✅ `parallel` | — |
| **(IA)³** | 1e-2 | 极低 | **0**（可合并）| ✅ `ia3` | ✅ `IA3Config` |
| MAM Adapter | 1e-4 | 中 | +20% | — | — |
| K-Adapter | 1e-4 | 中 | +N×5% | ✅ Stack | — |
| MAD-X | 1e-4 | 中 | +10% | ✅ Stack | — |
| AdaMix | 1e-3 | 中 | +N× (train), 0 (merge) | — | — |

### 3. 一致性测试矩阵

| 方法 | 测试类型 | 状态 |
|------|---------|------|
| Houlsby | 强一致 (minimal vs lib) | ✅ 完美匹配 608,640 |
| Pfeiffer | 强一致 | ✅ 完美匹配 304,320 |
| AdapterFusion | 弱一致 | ✅ minimal/lib 差 0.09% |
| AdapterDrop | 单元测试 | ✅ k=5 时 7 层 active |
| Compacter | PHM 数值 + 总参 | ✅ Kronecker 数值精确 |
| Parallel | 强一致 + 串/并联差异 | ✅ 304,320 match |
| **(IA)³** | ⭐ **三轨强一致** | ✅ **三方都是 55,296，初始 Δ=0** |
| MAM Adapter | 单元 (3 组件) | ✅ Prefix + Parallel 独立 |
| K-Adapter | 冻结切换 | ✅ N=2 冻 factual 后 304K |
| MAD-X | 语言切换 | ✅ 3 lang 输出不同 |
| AdaMix | 路由 + merge | ✅ stochastic train, det infer, merge N→1 |

### 4. 工程选型决策树

```
你的问题是什么？
│
├─ 极致省参 (<0.05%)？        → (IA)³ ⭐⭐⭐
├─ Few-shot 学习？            → (IA)³ ⭐⭐
├─ 多任务组合？               → AdapterFusion ⭐⭐
├─ 跨语言迁移？               → MAD-X ⭐⭐⭐
├─ 多领域知识注入？            → K-Adapter ⭐⭐
├─ 推理时延敏感？              → (IA)³（可合并）/ 用 LoRA
├─ 大模型 + 多 adapter 复用？   → MAD-X / AdaMix
├─ 追求 SOTA？                 → MAM Adapter ⭐⭐⭐
├─ Stable Diffusion？          → 用 LoKr (LoRA 系)
└─ 默认选择？                   → 用 LoRA（adapter 系已被 LoRA 超越）
```

## 跨专题 28 方法综合表

| 主线 | 方法数 | 切入点 | 典型方法 | 推理时延 |
|------|--------|-------|---------|---------|
| **Prompt-based** | 5 | 输入端（soft token）| Prompt, Prefix, P-Tuning v1/v2 | 有（占 context）|
| **Weight-based** | 12 | 权重端（低秩/量化）| LoRA, QLoRA, DoRA, VeRA... | 0（可合并）|
| **Adapter-based** | 11 | 结构端（加层）| Houlsby, (IA)³, MAM, MAD-X... | 有（除 (IA)³）|

### 跨专题工程选型（meta-comparison）

| 问题 | 推荐 |
|------|------|
| **65B 大模型 + 24GB GPU** | QLoRA ⭐⭐⭐ |
| **NLU 分类（小模型）** | P-Tuning v2 / LoRA |
| **NER 序列标注** | ⭐ P-Tuning v2（独家）|
| **NLG 生成** | LoRA / DoRA / Prefix Tuning |
| **极致省参（< 0.05%）** | ⭐ (IA)³ / VeRA |
| **不占 context window** | LoRA / Adapter 主线 |
| **可合并 0 推理时延** | LoRA / PiSSA / DoRA / (IA)³ |
| **多任务（千用户）** | VeRA / MAD-X / AdapterFusion |
| **多模态（视觉+语言）** | ⭐⭐⭐ LLaMA-Adapter / Q-Former / LLaVA |
| **追求 SOTA** | DoRA + MAM 组合 |
| **科研 baseline** | LoRA r=8（默认）|

## 全专题学习目标自测

读完本专题后，你应该能回答：

1. **公式题**：写出 Houlsby Adapter 公式 $\text{Adapter}(x) = x + W_{up}\sigma(W_{down}x)$ 的各项含义
2. **公式题**：解释为什么 $W_{up}$ 零初始化是关键（与 LoRA 的 $B$ 零初始化对比）
3. **公式题**：推导 Compacter 的 PHM 参数量 $n^3 + d_{out}d_{in}/n$
4. **公式题**：证明 Parallel Adapter (σ=identity) = LoRA
5. **公式题**：证明 Prefix Tuning ≡ Parallel Adapter（attention 等价分解）
6. **设计题**：MAM Adapter 为什么选 Prefix(attn) + Parallel(FFN)？
7. **设计题**：(IA)³ 为什么不学 $\ell_q$？
8. **对比题**：Houlsby vs Pfeiffer 实验上几乎相当，为什么社区主用 Pfeiffer？
9. **对比题**：AdapterFusion vs AdaMix 都是"多 adapter 组合"，本质差异？
10. **实践题**：在 LLaMA-7B 上估算 K-Adapter (5 类知识) 的参数量。
11. **应用题**：解释 Adapter 在 2023 后多模态时代如何复活（LLaMA-Adapter / Q-Former）。
12. **统一题**：写出 28 方法的统一公式 $h \leftarrow h + f(W_{down} \cdot x) \cdot W_{up}$，并归约任意 3 个方法。

## Git 里程碑

| Tag | Commit | 内容 |
|-----|--------|------|
| `adapter-base` | L1 | Houlsby + Pfeiffer (强一致 608K/304K) |
| `adapter-multitask` | L2-L3 | AdapterFusion + Drop + Compacter |
| `adapter-structure` | L4-L5 | Parallel + (IA)³ 三轨完美一致 |
| `adapter-unified` | L6 | MAM Adapter (理论高峰) |
| `adapter-app` | L7-L8 | K-Adapter + MAD-X + AdaMix |
| `adapter-family-complete` | _本次_ | L9-L10 + README |

## PEFT 三专题总结

```
你已完成 PEFT 三大主线学习:

  Prompt-tuning-family   (5 方法,  ~6 hours)
  LoRA-family            (12 方法, ~10 hours)
  Adapter-tuning-family  (11 方法, ~13 hours)
  -------------------------------------------
  总计                   28 方法,  ~29 hours

接下来推荐:
  ⭐⭐⭐ 对齐专题 (RLHF / DPO / SimPO) — 接续书本 Ch3
  ⭐⭐  长上下文 (LongLoRA / YaRN)
  ⭐⭐  MoE (Mixtral / DeepSeek)
  ⭐   推理优化 (vLLM / FlashAttention)
```

## 实用入口

| 我想 | 看这个 |
|------|--------|
| 快速入门 Adapter | [`lectures/01-houlsby-pfeiffer.md`](lectures/01-houlsby-pfeiffer.md) |
| 理解统一视角 | ⭐ [`lectures/06-mam-adapter.md`](lectures/06-mam-adapter.md) |
| 看三线综合 | ⭐ [`lectures/09-three-line-unification.md`](lectures/09-three-line-unification.md) |
| 看 PEFT 路线图 | [`lectures/10-peft-next-step.md`](lectures/10-peft-next-step.md) |
| 跑跑代码 | [`notebooks/`](notebooks/) 任选一个 |
| 跑全部测试 | `python -m pytest learning/adapter-tuning-family/src/tests/` |
| 验证环境 | `python learning/adapter-tuning-family/environment/verify_env.py` |
