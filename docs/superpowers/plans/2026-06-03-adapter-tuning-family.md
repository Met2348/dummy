# Adapter Tuning Family Implementation Plan

**Goal**: 完整实现 Adapter 家族 11 种方法的学习包（10 lecture + 11 minimal + 8 adapters + 1 peft + 10 notebook + 11 tests）

**Architecture**: 与 LoRA/Prompt 家族同构。三轨代码（minimal + adapters + peft 仅 IA3）。10 lectures = 8 主线 + 2 capstone。

**Tech Stack**: torch 2.13.0.dev cu130 / transformers 4.57.6 / adapters 1.3 / peft 0.19 / bitsandbytes 0.49

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/adapter-tuning-family/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- 11 个依赖 + 版本约束（adapters 强制 transformers 4.x）

### Task 1.3: environment/verify_env.py
- Part A 基础 / Part B GPU / Part C adapters smoke test

### Task 1.4: src/common.py
- 复用 LoRA 系列的 freeze/get_in_out_dims/target_linear_modules
- 新增 adapter 相关 helper: `insert_adapter_after_module`

### Task 1.5: papers/ 11 个占位 + README index

### Commit: `chore: adapter family scaffold`

---

## Phase 2: L1 Houlsby + Pfeiffer

### Task 2.1: lectures/01-houlsby-pfeiffer.md (25 slides)
- Adapter 范式起源、Houlsby 双串联结构、Pfeiffer 简化、参数量分析、与 LoRA 关系

### Task 2.2: src/houlsby_minimal.py
- `HoulsbyAdapter(d, r=16, act="gelu")`: down→act→up→+residual
- `HoulsbyGPT2`: 在 attn 后 + FFN 后各插一个

### Task 2.3: src/pfeiffer_minimal.py
- `PfeifferAdapter`: 同 Houlsby 但只在 FFN 后插
- 参数减半

### Task 2.4: src/houlsby_adapters.py + src/pfeiffer_adapters.py
- 用 adapters 库的 `AdapterConfig.load("houlsby" | "pfeiffer")`

### Task 2.5: src/tests/test_houlsby_consistency.py + test_pfeiffer_consistency.py
- 参数量、初始 forward = base、mini training

### Task 2.6: notebooks/01-houlsby-pfeiffer.ipynb
- 参数布局、minimal vs adapters 库、Houlsby vs Pfeiffer 双方对比 mini training

### Commit + Tag: `adapter-base`

---

## Phase 3: L2 AdapterFusion

### Task 3.1: lectures/02-adapterfusion.md (24 slides)
- 多任务挑战、两阶段训练、attention 融合机制、KQV 矩阵作用

### Task 3.2: src/adapterfusion_minimal.py
- `FusionLayer`: 多个冻结 adapter + 1 个 Q/K/V attention 融合层

### Task 3.3: src/adapterfusion_adapters.py
- 用 adapters 库的 add_adapter_fusion API

### Task 3.4: src/tests/test_adapterfusion.py
- 融合层参数量、3-task 玩具实验

### Task 3.5: notebooks/02-adapterfusion.ipynb
- 2 阶段流程图、3 任务玩具 demo

### Commit (在 multitask tag 里)

---

## Phase 4: L3 AdapterDrop + Compacter

### Task 4.1: lectures/03-adapterdrop-compacter.md (28 slides)
- AdapterDrop 推理加速、Compacter 的 PHM 数学（n=2 手算）、跨层共享

### Task 4.2: src/adapterdrop_minimal.py
- `AdapterDrop`: 训练时随机丢、推理时丢前 k 层

### Task 4.3: src/compacter_minimal.py
- `PHMLinear(n, d_in, d_out)`: A_i (n×n) × B_i (d/n × d/n)，Kronecker 求和
- `CompacterAdapter`: 用 PHMLinear 替代普通 Linear
- 关键：跨层共享 A_i

### Task 4.4: src/adapterdrop_adapters.py + src/compacter_adapters.py
- adapters 库的 drop_prob 和 compacter config

### Task 4.5: tests
- AdapterDrop: 训练前后参数量、drop 效果
- Compacter: PHM 参数量 = O(n² + d²/n)，n=2 时 ~50% 压缩 vs Houlsby

### Task 4.6: notebooks/03-adapterdrop-compacter.ipynb
- PHM 数学可视化、Compacter vs Houlsby 参数量对比

### Commit + Tag: `adapter-multitask`

---

## Phase 5: L4 Parallel Adapter

### Task 5.1: lectures/04-parallel-adapter.md (22 slides)
- "Towards a Unified View" 论文核心、串联 vs 并联、scaling factor

### Task 5.2: src/parallel_minimal.py
- `ParallelAdapter`: x → adapter_branch + base_branch → sum
- 与 LoRA 结构相似度（无非线性时 = LoRA）

### Task 5.3: src/parallel_adapters.py
- adapters 库的 ParallelConfig

### Task 5.4: tests + notebook
- Parallel vs Houlsby vs LoRA 三方对比

### Commit (在 structure tag 里)

---

## Phase 6: L5 (IA)³

### Task 6.1: lectures/05-ia3.md (24 slides)
- 3 个对角缩放向量 (l_k, l_v, l_ff)、为什么这么少够用、可合并性

### Task 6.2: src/ia3_minimal.py
- `IA3Linear`: x → base(x) * l (element-wise)
- 应用在 attn k/v 和 FFN 中间层

### Task 6.3: src/ia3_adapters.py + src/ia3_peft.py
- **三轨对照**：minimal vs adapters vs peft.IA3Config
- 验证三方 forward 一致

### Task 6.4: tests/test_ia3_three_way.py
- 三方参数量一致、forward 强一致

### Task 6.5: notebooks/05-ia3.ipynb
- 极致压缩展示（< 0.01% 参数）、(IA)³ vs VeRA 跨专题对比

### Commit + Tag: `adapter-structure`

---

## Phase 7: L6 MAM Adapter

### Task 7.1: lectures/06-mam-adapter.md (28 slides)
- "Towards a Unified View" 核心、Prefix=Parallel Adapter (k=v 注入)、LoRA=无激活 Adapter
- MAM 公式：m_attn (Prefix-like) + m_ffn (Parallel Adapter)

### Task 7.2: src/mam_minimal.py
- `MAMAdapter`: Prefix-style attention 注入 + Parallel FFN adapter
- 拆成 3 个独立 module

### Task 7.3: tests
- 各组件单独测试 + 组合测试

### Task 7.4: notebooks/06-mam-adapter.ipynb
- 统一视角可视化、Prefix ≡ Parallel Adapter 数学等价演示

### Commit + Tag: `adapter-unified`

---

## Phase 8: L7 K-Adapter + MAD-X

### Task 8.1: lectures/07-k-adapter-mad-x.md (26 slides)
- K-Adapter: 注入 factual / linguistic 知识，独立训练后 plug-in
- MAD-X: lang adapter (language-specific) + task adapter (task-specific)
- invertible adapter 概念

### Task 8.2: src/k_adapter_minimal.py
- 多个独立 adapter (factual_adapter, linguistic_adapter)
- toy knowledge: 10 条 (entity, relation, entity) 三元组训练数据

### Task 8.3: src/madx_minimal.py
- lang_adapter (3 种语言玩具) + task_adapter
- invertible_adapter（核心创新）

### Task 8.4: src/k_adapter_adapters.py + src/madx_adapters.py
- adapters 库对应 API

### Task 8.5: tests + notebook

### Commit (在 app tag 里)

---

## Phase 9: L8 AdaMix (MoE 路由)

### Task 9.1: lectures/08-adamix.md (24 slides)
- Mixture of Adapters、Stochastic routing during training、Average inference
- 与 MoE 区别（adapter 级而非 layer 级）

### Task 9.2: src/adamix_minimal.py
- `AdaMixLayer`: N 个 adapter expert + 随机 routing (训练) / averaging (推理)
- 训练时随机选 1 个 expert，推理时取平均

### Task 9.3: tests + notebook

### Commit + Tag: `adapter-app`

---

## Phase 10: L9 + L10

### Task 10.1: lectures/09-three-line-unification.md (28 slides)
- MAM Adapter 统一公式回顾
- Prefix Tuning ≡ Parallel Adapter (k=v 注入) 数学证明
- LoRA ≡ Adapter (无激活、低秩) 数学证明
- 28 方法横向对比 (5+12+11)
- 工程选型决策树（多场景）

### Task 10.2: lectures/10-peft-next-step.md (26 slides)
- Adapter 的"死亡": 2022 后为何无新方法（5 原因）
- Adapter 的"多模态复活":
  - LLaMA-Adapter v1/v2 (2023): 指令微调 + 多模态
  - Q-Former (BLIP-2): 跨模态 adapter
  - LLaVA projector: 最简 adapter
  - AdapterSoup: 权重平均
- 后续专题路线图: 长上下文 / 对齐 / MoE / 推理优化

### Task 10.3: notebooks/09-unification.ipynb
- 三线统一公式可视化
- Prefix=Parallel Adapter 数值验证

### Task 10.4: notebooks/10-peft-next-step.ipynb
- LLaMA-Adapter zero-init attention demo
- Q-Former pseudo-code walkthrough

### Commit (在 family-complete tag 里)

---

## Phase 11: README + 收官

### Task 11.1: learning/adapter-tuning-family/README.md
- 专题概览、学习路径、目录结构
- 横向对比表（11 方法）
- 跨专题 28 方法表
- 三线统一摘要
- 学习目标自测（10 题）
- Git 里程碑表

### Task 11.2: 最终 commit + tag `adapter-family-complete`

### Task 11.3: 最终交付报告

---

## 执行原则

1. **每个 phase 完成后立即 commit**（无中间 review）
2. **代码 + 测试 + notebook 并行写**（同一 phase 内）
3. **lecture 用模板快速生成**（不追求每篇都精雕细琢）
4. **notebook 优先用结构展示而非长 training**（保持 < 1 min/cell）
5. **tests 强制通过**（任何 fail 立即修）
6. **README 最后写**（避免反复修改）
