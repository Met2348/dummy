# Scaling Infra 学习专题 — 设计文档

> **承接**: 专题 1-5 全部架构 + 数据 ready
> **本专题**: Module 3 第 6 站 — 训练 infra（ZeRO/FSDP/Megatron/3D 并行）
> **战略地位**: 真训百亿模型的工程地基
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

数据 + 架构 ready 后，下一步是 **怎么训得起来**。本专题覆盖混合精度、梯度累积、ZeRO 三阶段、FSDP、Tensor/Pipeline 并行、3D 并行、Megatron-Core 全套。

### 1.1 为什么单独成专题

- PEFT/RL 用 LoRA + 单卡，避开了大部分 infra 问题
- 但真预训练 270M+ 必须用 FSDP/ZeRO（即使单卡）
- 7B+ 必须 3D 并行
- 不学 infra → 无法读懂 Megatron / Llama-3 / DeepSeek-V3 训练 paper

### 1.2 本专题的"硬核度"

- **算法 30%**: ZeRO 三阶段分片数学、3D 并行通信复杂度
- **工程 60%**: FSDP wrap / DeepSpeed config / NCCL 通信
- **理论 10%**: scaling laws / Chinchilla / μP

---

## 2. 方法清单（16 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **Scaling Laws (Kaplan)** | 2020 | OpenAI | loss vs params/data/compute |
| 2 | **Chinchilla Laws** | 2022 | Hoffmann | 平衡 params 和 data |
| 3 | **μP (Maximal Update Param)** | 2022 | Yang | 超参 transfer ⭐ |
| 4 | **Mixed Precision FP16** | 2017 | Micikevicius | loss scale |
| 5 | **BF16** | 2018+ | — | TPU 起源 / Ampere 后通用 |
| 6 | **FP8** | 2022+ | NVIDIA Hopper | H100 训练 |
| 7 | **Gradient Accumulation** | — | — | 显存换算力 |
| 8 | **Gradient Checkpointing** | 2016 | Chen | 重算换显存 |
| 9 | **ZeRO Stage 1/2/3** | 2019-2020 | Microsoft | optim/grad/param 分片 ⭐ |
| 10 | **PyTorch FSDP** | 2022 | Meta | DDP + ZeRO 集大成 ⭐ |
| 11 | **Tensor Parallel (Megatron)** | 2019 | NVIDIA | column/row split |
| 12 | **Pipeline Parallel (GPipe)** | 2018 | Google | 1F1B / interleaved |
| 13 | **3D Parallel** | 2021+ | — | DP × TP × PP |
| 14 | **DeepSpeed ZeRO-Infinity** | 2021 | Microsoft | offload to NVMe |
| 15 | **Megatron-Core** | 2024 | NVIDIA | 工业级框架 |
| 16 | **NCCL 通信原语** | — | NVIDIA | all-reduce / all-gather |

---

## 3. Lecture 结构（14 篇 = 13 主线 + 1 capstone）

| Lecture | 主方法 | 时长 |
|---------|--------|------|
| **L01** Scaling Laws | Kaplan + Chinchilla | 75 min |
| **L02** Chinchilla 教训 | 数据/参数平衡 | 60 min |
| **L03** μP 超参 transfer | Yang 2022 | 90 min |
| **L04** Mixed Precision | FP16 / BF16 / FP8 | 75 min |
| **L05** Grad Accum + Ckpt | 显存换算力 | 60 min |
| **L06** ZeRO Stage 1/2/3 ⭐ | Microsoft | 120 min |
| **L07** PyTorch FSDP ⭐ | Meta | 120 min |
| **L08** Tensor Parallel | Megatron | 90 min |
| **L09** Pipeline Parallel | GPipe / 1F1B | 90 min |
| **L10** 3D Parallel | DP × TP × PP 组合 | 90 min |
| **L11** DeepSpeed 完整栈 | ZeRO + offload + infinity | 75 min |
| **L12** Megatron-Core | 工业级框架 | 60 min |
| **L13** 通信原语 | NCCL all-reduce/gather | 60 min |
| **L14** Capstone：FSDP 训 350M | 单卡 + 模拟多卡 | 120 min |

**总学时**: 14 lecture × 平均 85 min + 8h notebook ≈ 18 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（显存 / 速度 瓶颈）
## Slide 3-6: 算法 / 通信模式
## Slide 7-10: 显存 / 带宽 / 算力 分析
## Slide 11-15: 代码逐行（FSDP wrap / DeepSpeed config）
## Slide 16-20: 实验：1 卡 vs 4 卡 vs 16 卡
## Slide 21-24: 工程坑 + 调试
## Slide 25-26: 思考题
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 | 工业 |
|------|-------------|-----|------|
| 混合精度 | ✅ | ✅ torch.amp | (deepspeed) |
| Grad Accum | ✅ | ✅ | — |
| Grad Ckpt | — | ✅ torch.utils.checkpoint | — |
| ZeRO-1 | ✅ 手写玩具 | ✅ deepspeed | ✅ deepspeed |
| ZeRO-2/3 | — | ✅ deepspeed | ✅ |
| FSDP | ✅ wrap 教学 | ✅ torch FSDP | ✅ |
| Tensor Parallel | ✅ column/row 玩具 | — | ✅ Megatron-Core |
| Pipeline Parallel | ✅ GPipe 玩具 | — | ✅ Megatron-Core |
| 3D Parallel | — | — | ✅ Megatron-Core |
| ZeRO-Infinity | — | ✅ deepspeed | — |
| NCCL primitives | ✅ benchmark | — | — |

---

## 6. 一致性测试

```python
def test_grad_accum_equivalence():     # accum=8 step1 vs batch×8 single step
def test_mixed_precision_loss_match():  # fp16/bf16 vs fp32 < 0.1 ppl diff
def test_fsdp_vs_ddp_step():            # 同 seed 单步 loss < 1e-4
def test_zero_naive_shard():            # 手写 ZeRO-1 vs deepspeed < 1e-4
def test_tp_column_row_correctness():    # split-merge 结果一致
def test_flops_calculator_accuracy():    # 预测 vs nvidia-smi 实测 < 10%
```

---

## 7. Notebook 结构（14 个）

每个 lecture 一个 ipynb：
1. import + 多 GPU 环境
2. 算法 / 通信图解
3. minimal 实现
4. 库对照（FSDP/DeepSpeed）
5. 性能 benchmark（显存 / 速度 / 吞吐）
6. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (WSL2 + 推荐 ≥2 卡)
torch>=2.5+cu130
deepspeed>=0.15
megatron-core>=0.9
nvidia-nccl-cu12  # NCCL
flash-attn>=2.6
einops
```

**verify_env.py 三段式**:
- Part A: 基础（torch + deepspeed + megatron-core）
- Part B: GPU + NCCL + sm_120
- Part C: FSDP 2-rank smoke（torchrun --nproc-per-node=2 用 CPU shard）

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `infra-scaling-laws` | L01-L03: Scaling laws + μP | 3 |
| `infra-precision` | L04-L05: AMP + grad accum/ckpt | 2 |
| `infra-zero` | L06: ZeRO 1/2/3 | 3 |
| `infra-fsdp` | L07: FSDP | 3 |
| `infra-parallel` | L08-L10: TP + PP + 3D | 4 |
| `infra-frameworks` | L11-L13: DeepSpeed + Megatron + NCCL | 3 |
| `scaling-infra` | L14: Capstone + README | 3 |

---

## 10. 跨专题衔接

### 上游
- 专题 2 transformer-deep：80M GPT-mini 作 baseline

### 下游
- 专题 7 预训练：用 FSDP 训 270M
- 专题 8 graduation：理解 DeepSeek-V3 训练栈

### 跨专题对照表预留位
| 训练规模 | 推荐方案 | 显存/卡 |
|---------|---------|--------|
| <1B | 单卡 + grad ckpt | 24GB |
| 1B-7B | FSDP 4-8 卡 | 4×40GB |
| 7B-70B | 3D 并行 | 32×80GB |
| 70B+ | Megatron-Core 集群 | 集群 |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 单卡演示价值有限 | 高 | 中 | torchrun --nproc-per-node=2 模拟 + 云租用建议 |
| Megatron-Core 安装坑 | 高 | 高 | Docker 备份 + apex 版本严格 |
| FSDP wrap policy 易错 | 中 | 中 | 提供 known-good wrap example |
| NCCL 通信 hang | 中 | 高 | 提供 NCCL_DEBUG=INFO 教程 |
| FP8 需 H100 | 高 | 中 | 仅展示代码 + benchmark 数字 |
| μP 数学难 | 中 | 低 | L03 配合验证脚本 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-kaplan-2020-scaling-laws.md
├── 02-hoffmann-2022-chinchilla.md
├── 03-yang-2022-mup.md
├── 04-micikevicius-2017-mixed-precision.md
├── 05-fp8-2022.md
├── 06-chen-2016-grad-ckpt.md
├── 07-rajbhandari-2019-zero.md
├── 08-meta-2022-fsdp.md
├── 09-shoeybi-2019-megatron.md
├── 10-huang-2018-gpipe.md
├── 11-rajbhandari-2021-zero-infinity.md
├── 12-megatron-core-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-scaling-infra.md` 的 7 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L03 Scaling + μP（tag `infra-scaling-laws`）
- Phase 3: L04-L05 精度 + accum（tag `infra-precision`）
- Phase 4: L06 ZeRO（tag `infra-zero`）
- Phase 5: L07 FSDP（tag `infra-fsdp`）
- Phase 6: L08-L10 并行三件套（tag `infra-parallel`）
- Phase 7: L11-L14 框架 + Capstone（tag `infra-frameworks` + `scaling-infra`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
