# SSM / Hybrid 学习专题 — 设计文档

> **承接**: 专题 2 transformer-deep（提供 80M GPT-mini 作 Transformer baseline）
> **本专题**: Module 3 第 4 站 — 非 Transformer 路线全图（Mamba / RWKV / Hybrid）
> **战略地位**: 2024-2026"另一条路"，长上下文 + 推理速度优势明显
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

State Space Models (SSM) 从 2021 S4 → 2024 Mamba-2 → 2025 Mamba-3 / RWKV-7 / Jamba 走出了"非 attention"路线。理解 SSM 不是为了立刻替代 Transformer，而是：
1. 把握下一代架构的可能性
2. 长上下文场景（≥1M）的工程优势
3. Hybrid（Jamba/Zamba）= 主流商业方向

### 1.1 为什么需要专门一专题

- Mamba 数学完全独立于 attention（状态空间方程）
- 训练 / 推理代码栈完全不同（mamba-ssm / causal-conv1d）
- 2025 商业 Jamba-1.5 + Mistral Codestral-Mamba 已经实用

### 1.2 与 Transformer 的关系

- **互补不替代**：Mamba 强 sequence-level，attention 强 retrieval
- **混合架构是赢家**：Jamba (Mamba+attn+MoE)、Zamba (Mamba+共享 attn)

---

## 2. 方法清单（12 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **HiPPO** | 2020 | Gu | 多项式投影 / 历史压缩 |
| 2 | **S4** | 2021 | Gu | 卷积形式 / 频域 / 离散化 |
| 3 | **S5** | 2022 | Smith | 简化 S4 |
| 4 | **Mamba (S6)** | 2023.12 | Gu+Dao | Selective SSM + 硬件 scan ⭐ |
| 5 | **Mamba-2 (SSD)** | 2024.05 | Dao+Gu | 矩阵分解 / 与 attention 等价 ⭐ |
| 6 | **Mamba-3** | 2025 | — | 长上下文优化 |
| 7 | **RWKV-7** | 2025 | Peng | linear attention 路线 |
| 8 | **RetNet** | 2023 | Microsoft | retention mechanism |
| 9 | **Jamba** | 2024 | AI21 | Mamba+attn+MoE 混合 ⭐ |
| 10 | **Zamba / Zamba-2** | 2024 | Zyphra | Mamba + 共享 attn |
| 11 | **Codestral-Mamba** | 2024 | Mistral | Mamba 代码模型 |
| 12 | **selective scan kernel** | 2023-2024 | — | 硬件感知实现 |

---

## 3. Lecture 结构（11 篇 = 10 主线 + 1 capstone）

| Lecture | 主方法 | 时长 |
|---------|--------|------|
| **L01** SSM 数学背景 | HiPPO / 状态空间方程 | 90 min |
| **L02** S4 / S5 | 卷积形式 + 离散化 | 90 min |
| **L03** Mamba ⭐ | Selective SSM + scan kernel | 120 min |
| **L04** Mamba-2 ⭐ | SSD / 与 attention 等价 | 90 min |
| **L05** Mamba-3 | 长上下文优化 | 60 min |
| **L06** RWKV-7 | linear attention 路线 | 75 min |
| **L07** RetNet / RWKV-6 | retention 机制 | 60 min |
| **L08** Jamba ⭐ | Mamba+attn+MoE 混合 | 90 min |
| **L09** Zamba / Codestral-Mamba | 其他 hybrid | 60 min |
| **L10** Hybrid 设计 | 何时混 / 比例 / layer 选择 | 75 min |
| **L11** Capstone：130M mini-Mamba | OpenWebText 1B + 对照 GPT-mini | 120 min |

**总学时**: 11 lecture × 平均 75 min + 5h notebook ≈ 12 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（解决什么）
## Slide 3-6: 状态空间方程 / 算法
## Slide 7-10: 与 attention 对比
## Slide 11-14: 硬件实现（scan kernel）
## Slide 15-18: 代码逐行
## Slide 19-22: 实验：长上下文 / 速度 / ppl
## Slide 23-25: 思考题 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 (mamba-ssm) | 工业 |
|------|-------------|---------------|------|
| S4 卷积形式 | ✅ | — | — |
| Mamba block | ✅ 无 selective scan | ✅ mamba_ssm | — |
| Mamba-2 SSD | ✅ | ✅ | — |
| RWKV-7 block | ✅ 简化 | ✅ rwkv | — |
| Jamba layer | ✅ hybrid 配比 | — | — |
| 130M Mamba 完整 | ✅ | — | — |

---

## 6. 一致性测试

```python
def test_s4_convolutional_vs_recurrent():    # 同 input 两种形式输出一致
def test_mamba_naive_vs_lib():                # mamba-ssm 库 < 1e-4
def test_mamba2_vs_attention_equivalence():   # 特定情形 SSD ≈ attention
def test_mamba_long_extrapolation():          # 训 1k 推 4k ppl 合理
def test_jamba_layer_routing():               # mixed layer 输出 shape 对
def test_mini_mamba_train_loss():             # 500 step loss 下降合理
```

---

## 7. Notebook 结构（11 个）

每个 lecture 一个 ipynb：
1. import + 模型加载
2. 状态空间可视化
3. minimal 实现
4. mini 训练（500 step）
5. 库对照
6. 长上下文外推测试
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (WSL2)
torch>=2.5+cu130
mamba-ssm>=2.2
causal-conv1d>=1.4
rwkv>=0.8
einops
```

**verify_env.py 三段式**:
- Part A: 基础（torch + mamba-ssm + causal-conv1d）
- Part B: GPU + sm_120
- Part C: Mamba forward smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `ssm-foundations` | L01-L02: HiPPO + S4 | 2 |
| `ssm-mamba` | L03-L05: Mamba 1/2/3 | 4 |
| `ssm-rwkv` | L06-L07: RWKV / RetNet | 2 |
| `ssm-hybrid-arch` | L08-L10: Jamba / Zamba / 设计 | 3 |
| `ssm-hybrid` | L11: Capstone + README | 3 |

---

## 10. 跨专题衔接

### 上游
- 专题 2 transformer-deep：作 baseline 对比

### 下游
- 专题 5 长上下文：Mamba 天然长上下文优势
- 专题 7 预训练：可考虑训 Mamba 替代 Transformer

### 跨专题对照表预留位
| 架构 | sequence 复杂度 | 长上下文 | 训练成熟度 |
|------|---------------|---------|----------|
| Transformer | O(N²) | 中（YaRN）| ⭐⭐⭐⭐⭐ |
| Mamba | O(N) | 强 | ⭐⭐⭐⭐ |
| RWKV | O(N) | 强 | ⭐⭐⭐ |
| Jamba (hybrid) | O(N)+O(N²) | 强 | ⭐⭐⭐⭐ |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| mamba-ssm 安装 cuda 版本敏感 | 高 | 高 | 提供 known-good combo |
| 130M Mamba 1B token 可能 underfit | 中 | 中 | 用 reduced loss 比较 |
| 教学价值 > 实际性能 | 高 | 低 | 明确"理解为主"定位 |
| selective scan kernel 难调 | 中 | 中 | 提供 naive scan 教学 + 库对照 |
| Jamba hybrid 比例选择经验少 | 中 | 中 | 复用 AI21 配比（1:7 attn:mamba）|

---

## 12. 论文 / 资料占位

```
papers/
├── 01-gu-2020-hippo.md
├── 02-gu-2021-s4.md
├── 03-smith-2022-s5.md
├── 04-gu-2023-mamba.md                # 原论文
├── 05-dao-2024-mamba2.md
├── 06-mamba3-2025.md
├── 07-peng-2025-rwkv7.md
├── 08-sun-2023-retnet.md
├── 09-lieber-2024-jamba.md            # AI21
├── 10-zamba2-2024.md
├── 11-codestral-mamba-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-ssm-hybrid.md` 的 5 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L02 SSM 基础（tag `ssm-foundations`）
- Phase 3: L03-L05 Mamba 1/2/3（tag `ssm-mamba`）
- Phase 4: L06-L07 RWKV / RetNet（tag `ssm-rwkv`）
- Phase 5: L08-L11 Hybrid + Capstone（tag `ssm-hybrid-arch` + `ssm-hybrid`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
