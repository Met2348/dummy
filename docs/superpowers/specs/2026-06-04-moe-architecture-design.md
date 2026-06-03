# MoE Architecture 学习专题 — 设计文档

> **承接**: 专题 2 transformer-deep（提供 80M GPT-mini 作为 dense baseline）
> **本专题**: Module 3 第 3 站 — MoE 架构演化全图
> **战略地位**: 2024-2026 开源 SOTA 几乎全是 MoE（DeepSeek-V3 / Mixtral / Phi-MoE / Qwen3-MoE）
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

MoE（Mixture of Experts）从 Shazeer 2017 到 DeepSeek-V3 2024 走过了 7 年。本专题串起完整路由算法演化 + 训练稳定化（aux loss / z-loss / Aux-Free）+ 推理效率（expert offload / grouped GEMM）。

### 1.1 为什么 MoE 是 Module 3 高峰之一

- DeepSeek-V3 671B 用 37B 激活实现 GPT-4 水平 → MoE 让"算力等效"成立
- Mixtral / Phi-MoE / Qwen3-MoE 全是开源 MoE
- 不学 MoE → 看不懂 2024+ 大部分开源新模型

### 1.2 本专题的"硬核度"

- **路由算法 50%**: top-1/top-2 / Expert Choice / Aux-Free 偏置
- **训练稳定 30%**: aux loss / z-loss / capacity factor / 路由崩塌
- **infra 工程 20%**: grouped GEMM / expert parallel / offload

---

## 2. 方法清单（14 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **Shazeer Sparse MoE** | 2017 | Shazeer | MoE 起源（LSTM 时代）|
| 2 | **GShard** | 2020 | Lepikhin | top-2 routing + expert parallel |
| 3 | **Switch Transformer** | 2021 | Fedus | top-1 简化 + 1.6T |
| 4 | **Expert Choice** | 2022 | Zhou | experts 选 tokens（反向路由）|
| 5 | **ST-MoE** | 2022 | Zoph | router z-loss 稳定 |
| 6 | **Mixtral 8x7B** | 2023 | Mistral | 开源 MoE 起点 |
| 7 | **DeepSeekMoE** | 2024 | DeepSeek-V2 | 细粒度 + 共享专家 |
| 8 | **Aux-Loss-Free 路由** | 2024 | DeepSeek-V3 | 偏置项替代 aux loss ⭐ |
| 9 | **Phi-3.5-MoE** | 2024 | Microsoft | 小 MoE 路线 |
| 10 | **Qwen3-MoE** | 2025 | Alibaba | A3B / 235B 系列 |
| 11 | **MoR (Mixture of Recursions)** | 2025 | — | MoE × Recurrence |
| 12 | **Capacity Factor** | 2020+ | — | per-expert token cap |
| 13 | **Expert Offloading** | 2023+ | — | 推理时 CPU/Disk offload |
| 14 | **Grouped GEMM** | 2023+ | megablocks | 高效 sparse compute |

---

## 3. Lecture 结构（13 篇 = 12 主线 + 1 capstone）

| Lecture | 主方法 | 时长 |
|---------|--------|------|
| **L01** MoE 概念起源 | Shazeer 2017 | 60 min |
| **L02** GShard top-2 | Lepikhin 2020 | 75 min |
| **L03** Switch Transformer top-1 | Fedus 2021 | 75 min |
| **L04** Expert Choice 反向路由 | Zhou 2022 | 60 min |
| **L05** Mixtral 8x7B | Mistral 2023 | 60 min |
| **L06** DeepSeekMoE 细粒度 | DeepSeek 2024 | 75 min |
| **L07** Aux-Loss-Free ⭐ | DeepSeek-V3 2024 | 120 min |
| **L08** Phi-MoE 小 MoE | Microsoft 2024-25 | 60 min |
| **L09** Qwen3-MoE | Alibaba 2025 | 60 min |
| **L10** MoR 新方向 | 2025 | 60 min |
| **L11** MoE 训练稳定化 | router z-loss / capacity / 崩塌 | 90 min |
| **L12** MoE 推理优化 | expert offload + grouped GEMM | 90 min |
| **L13** Capstone：4-expert mini-MoE | 完整训练 + 对比 dense | 120 min |

**总学时**: 13 lecture × 平均 80 min + 7h notebook ≈ 16 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（这个路由解决什么）
## Slide 3-6: 路由算法（数学 + 伪代码）
## Slide 7-10: 直觉解释（负载均衡 / 专家专业化）
## Slide 11-15: 训练稳定（aux loss / z-loss）
## Slide 16-20: 代码逐行（minimal vs megablocks）
## Slide 21-25: 实验：dense vs MoE ppl / 路由热图
## Slide 26-28: 陷阱（崩塌 / 路由抖动）+ 思考题
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 (megablocks) | 工业 (DeepSpeed-MoE) |
|------|-------------|---------------|--------------------|
| top-2 router | ✅ | ✅ megablocks | ✅ DeepSpeed-MoE |
| top-1 (Switch) | ✅ | ✅ | ✅ |
| Expert Choice | ✅ 反向 | — | — |
| Aux-Loss-Free | ✅ 偏置项 ⭐ | — | — |
| router z-loss | ✅ | ✅ | ✅ |
| capacity factor | ✅ | ✅ | ✅ |
| grouped GEMM | — | ✅ | — |
| expert parallel | — | ✅ | ✅ |
| expert offload | ✅ 教学版 | — | — |
| 4-expert mini-MoE | ✅ | — | — |

---

## 6. 一致性测试

```python
def test_top2_router_load_balance():    # 训 100 step,各 expert 利用率 1/k ± 30%
def test_aux_free_stability():           # 100 step,偏置项收敛
def test_router_z_loss_decreases():      # z-loss 单调下降
def test_capacity_drop():                # 超 capacity 的 token 正确丢弃
def test_grouped_gemm_correctness():     # naive vs grouped < 1e-4
def test_expert_offload_equivalence():   # offload 前后输出一致
def test_mini_moe_vs_dense_ppl():        # MoE val ppl < dense × 0.9
```

---

## 7. Notebook 结构（13 个）

每个 lecture 一个 ipynb：
1. import + 模型加载
2. 路由算法可视化（路由概率热图）
3. minimal 实现 step-by-step
4. mini 训练（500 step）
5. 库对照（megablocks）
6. 路由健康度（利用率 / drop rate / z-loss）
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (WSL2)
torch>=2.5+cu130
flash-attn>=2.6
megablocks>=0.5
deepspeed>=0.15
einops
transformers>=5.0  # 加载 Mixtral / DeepSeek 对照
```

**verify_env.py 三段式**:
- Part A: 基础（torch + megablocks + deepspeed）
- Part B: GPU + sm_120
- Part C: top-2 MoE smoke（10 step）

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `moe-routing` | L01-L04: Shazeer / GShard / Switch / Expert Choice | 4 |
| `moe-modern` | L05-L06: Mixtral + DeepSeekMoE | 2 |
| `moe-aux-free` | L07: Aux-Loss-Free ⭐ | 2 |
| `moe-small` | L08-L09: Phi-MoE + Qwen3-MoE | 2 |
| `moe-new` | L10: MoR | 1 |
| `moe-stability` | L11-L12: 训练稳定 + 推理优化 | 3 |
| `moe-arch` | L13: Capstone + README | 3 |

---

## 10. 跨专题衔接

### 上游
- 专题 2 transformer-deep：80M GPT-mini 作 dense baseline

### 下游
- 专题 6 infra：MoE expert parallel 演示
- 专题 7 预训练：Phi-tiny 可考虑加 MoE 层
- 专题 8 graduation：理解 DeepSeek-V3 完整架构

### 跨专题对照表预留位
| MoE 模型 | active / total | 路由 | 备注 |
|---------|--------------|------|------|
| Mixtral 8x7B | 13B / 47B | top-2 | 开源经典 |
| DeepSeek-V3 | 37B / 671B | Aux-Free | 开源 SOTA |
| Phi-3.5-MoE | 6.6B / 41.9B | top-2 | 小 MoE |
| Qwen3-235B-A22B | 22B / 235B | — | 通用 |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| megablocks 安装坑 | 高 | 高 | Docker + known-good combo |
| expert parallel 需 ≥2 卡 | 高 | 中 | 单卡 4 expert 演示 |
| 路由崩塌（1 expert 拿全部）| 中 | 高 | router z-loss + capacity factor + 监控 |
| Aux-Free 偏置项不收敛 | 中 | 高 | 严格按论文 update_rate=1e-3 + 偏置 init 0 |
| MoE 训练比 dense 慢 30% | 中 | 中 | grouped GEMM 优化 + benchmark |
| 4-expert 玩具 ppl 收益不显著 | 中 | 中 | 加大 expert num 或缩 dense baseline |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-shazeer-2017-sparse-moe.md
├── 02-lepikhin-2020-gshard.md
├── 03-fedus-2021-switch.md
├── 04-zhou-2022-expert-choice.md
├── 05-zoph-2022-stmoe.md             # router z-loss
├── 06-mixtral-2023.md
├── 07-deepseekmoe-2024.md            # V2
├── 08-deepseek-v3-2024.md            # Aux-Free
├── 09-phi-3.5-moe-2024.md
├── 10-qwen3-moe-2025.md
├── 11-mor-2025.md
├── 12-megablocks-2023.md             # grouped GEMM
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-moe-architecture.md` 的 7 个 Phase 推进：

- Phase 1: 基础设施（WSL2 切换）
- Phase 2: L01-L04 路由四方法（tag `moe-routing`）
- Phase 3: L05-L06 Mixtral + DeepSeekMoE（tag `moe-modern`）
- Phase 4: L07 Aux-Loss-Free（tag `moe-aux-free`）
- Phase 5: L08-L10 Phi/Qwen3/MoR（tag `moe-small` + `moe-new`）
- Phase 6: L11-L12 训练稳定 + 推理（tag `moe-stability`）
- Phase 7: L13 Capstone + README（tag `moe-arch`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
