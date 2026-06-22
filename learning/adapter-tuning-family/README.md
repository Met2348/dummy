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
│   ├── adapter_original_minimal.py                         # 原始 Adapter 概念（纯 Python，无 torch）
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
│   └── tests/                                            # 9 个测试文件
└── notebooks/                        # 10 个 ipynb
```

## 环境配置

> **重要警告**：本专题的 `adapters` 库 (AdapterHub) **强制依赖 transformers 4.x**，与本仓库统一环境（transformers 5.10.2）冲突。**本仓库 venv 故意不安装 `adapters`**——装它会把 transformers 降级到 4.x、破坏其它专题。
>
> - **手写 `*_minimal.py` + `peft` 的 `ia3_peft.py` 不依赖 `adapters`，在本仓库 venv 直接可跑**（即本专题的主验证路径，见下方「运行验证」）。
> - **9 个 `*_adapters.py`（AdapterHub 调包版）本地无法运行**：脚本在 `from adapters import ...` 处 fail-fast 报 `ModuleNotFoundError`（exit 1，非静默假成功）。仅作教学对照阅读；如需真跑，请**另建** transformers 4.x 独立 venv 后 `pip install adapters`。

```powershell
# 本仓库统一 venv（transformers 5.x）—— 验证手写 + peft 路径，无需 adapters：
.\.venv\Scripts\python.exe learning/adapter-tuning-family/environment/verify_env.py
```

预期输出（本仓库 venv，transformers 5.10.2）：

```
Part A (基础):           PASS    # torch cu13x, transformers 5.10.2, peft 已装; adapters [SKIP] not installed (optional)
Part B (GPU):            PASS    # RTX 3080 Ti Laptop 16GB, sm_86
Part C (adapters lib):   SKIP    # adapters 未装 → 跳过（minimal/peft 测试仍全跑）
```

> 若要复现书中「adapters 1.3 + transformers 4.57」的 Part C PASS 输出，需在独立 4.x venv 中运行；本仓库统一环境下 Part C 正常显示 SKIP。

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

## 运行验证（Runbook）

文档入口命令清单：[`runbook.yaml`](runbook.yaml)。用 audit harness 一键验证「照文档跑」是否真能跑通：

```bash
# V0（文档静态：脚本存在）+ V1（smoke：直跑 main 到完成）
.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --runbook \
  --modules adapter-tuning-family --timeout 300
```

**结果（RTX 3080 Ti Laptop 16GB，transformers 5.10.2）：V1 13/13 PASS；9 个 `*_adapters.py` 标 `tier: skip`（见下）。**

### 可本地直跑的入口（13 个，V1）

全部脚本**无 argparse**（直跑 `main()`，固定小预算 GPT-2，纯结构/参数量演示、**无梯度训练**），故 `v0: false`、无 smoke 参数——full 形态即 smoke 形态：

| 入口 | 命令 | 验证点 |
|------|------|--------|
| 原始 Adapter（纯 Python） | `python learning/adapter-tuning-family/src/adapter_original_minimal.py` | 手算 608,640 参数 + 近恒等输出（CPU 秒级）|
| Houlsby minimal | `python .../src/houlsby_minimal.py` | 608,640 参数，初始 forward=base（误差 0）|
| Pfeiffer minimal | `python .../src/pfeiffer_minimal.py` | 304,320 参数（单串联）|
| AdapterFusion minimal | `python .../src/adapterfusion_minimal.py` | 多任务 attention 融合 |
| AdapterDrop minimal | `python .../src/adapterdrop_minimal.py` | k=5 时 7 层 active |
| Compacter minimal | `python .../src/compacter_minimal.py` | PHM Kronecker 数值精确 |
| Parallel minimal | `python .../src/parallel_minimal.py` | σ=identity 等价 LoRA |
| (IA)³ minimal | `python .../src/ia3_minimal.py` | 55,296 参数，初始 Δ=0 |
| ⭐ (IA)³ **peft**（第三轨）| `python .../src/ia3_peft.py` | peft `IA3Config`，55,296（三轨一致）|
| MAM minimal | `python .../src/mam_minimal.py` | Prefix(attn)+Parallel(FFN) |
| K-Adapter minimal | `python .../src/k_adapter_minimal.py` | 多类知识 + 冻结切换 |
| MAD-X minimal | `python .../src/madx_minimal.py` | 语言 Stack 任务，跨语言切换 |
| AdaMix minimal | `python .../src/adamix_minimal.py` | 随机路由/平均推理/merge N→1 |

> 单跑前先 `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8`（避免 Windows 中文/emoji 输出乱码）。除「原始 Adapter」外均会加载 GPT-2 权重（首次联网下载、后续走缓存），单条 ~15–23s。

### 本地跳过的入口（9 个 `*_adapters.py`，`tier: skip`）

`houlsby_adapters` / `pfeiffer_adapters` / `adapterfusion_adapters` / `adapterdrop_adapters` / `compacter_adapters` / `parallel_adapters` / `ia3_adapters` / `k_adapter_adapters` / `madx_adapters`。

**原因**：需 AdapterHub `adapters` 库，与本仓库 transformers 5.x 冲突，故本地跳过（详见上方「环境配置」）。这些脚本在 import 处 fail-fast（`ModuleNotFoundError`，exit 1），不是静默假成功；要运行须另建 transformers 4.x venv。

### 关键坑注记

- **`adapters` 库不可在本仓库 venv 安装**——会把 transformers 降级到 4.x、破坏其它专题。需对照阅读 `*_adapters.py` 时，请另建独立 4.x 环境。
- **无 argparse**：脚本不接受 `--flag`，runbook 里 `v0: false`（跳过 `--help` 探针）。
- **`adapter_original_minimal.py` 0.x 秒 PASS 属正常**：纯 Python 数值 self-test（不加载模型），非 no-op——stdout 含真实参数量与近恒等校验。

### 测试（V2）

一致性单测（9 个文件，~127s）走 `--tests`（不进 runbook）：

```bash
.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --modules adapter-tuning-family --tests --timeout 600
# 或直接： python -m pytest learning/adapter-tuning-family/src/tests/
```
