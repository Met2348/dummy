# Long Context 学习专题 — 设计文档

> **承接**: 专题 2 transformer-deep（提供 RoPE / GQA / FA 基础）
> **本专题**: Module 3 第 5 站 — 长上下文方法论主线
> **战略地位**: 2024-2026 上下文从 8k → 1M → 10M，是产品分水岭
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

长上下文有两条主线：**RoPE scaling**（YaRN / NTK-aware）和 **分布式 attention**（Ring / Striped / Infini）。本专题覆盖两条线 + 评测（NIAH / RULER）+ 训练数据 + 工程 trick。

### 1.1 为什么单独成专题

- 2024-2026 商业模型核心竞争点：Gemini 2M / Claude 200k / GPT-4 128k
- 开源 ≤8k → ≥32k 必须懂 YaRN
- 1M+ 必须懂 Ring Attention 分布式
- "Lost in the middle"陷阱普遍存在 → 评测重要

### 1.2 与 Transformer 专题的关系

- Transformer 专题：vanilla attention + RoPE 基础
- 本专题：把 8k 扩到 1M 的所有 trick

---

## 2. 方法清单（14 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **Position Interpolation** | 2023.06 | Meta | 简单 RoPE scaling |
| 2 | **NTK-aware scaling** | 2023.07 | LocalLlama | base scaling |
| 3 | **YaRN** | 2023.09 | Peng | NTK by parts + attn temp ⭐ |
| 4 | **LongRoPE** | 2024 | Microsoft | search-based |
| 5 | **Ring Attention** | 2023.10 | Liu | sequence parallel 1M ⭐ |
| 6 | **Striped Attention** | 2024 | — | Ring 改进 |
| 7 | **Infini-Attention** | 2024 | Google | compressive memory |
| 8 | **FlashAttention long-context** | 2024 | — | block-sparse / window |
| 9 | **NoPE** | 2023 | — | 无位置编码 + extrapolation |
| 10 | **3D RoPE / Q-RoPE** | 2024 | — | 多模态 / 视频位置 |
| 11 | **NIAH** | 2023 | gkamradt | Needle in Haystack |
| 12 | **RULER** | 2024 | NVIDIA | 综合长上下文 benchmark |
| 13 | **Long-context 数据 packing** | 2024 | — | book/repo 长 sample 拼接 |
| 14 | **Lost in the middle** | 2023 | Liu (Stanford) | 中间位置注意力衰减 |

---

## 3. Lecture 结构（13 篇 = 12 主线 + 1 capstone）

| Lecture | 主方法 | 时长 |
|---------|--------|------|
| **L01** 长上下文 2024-2026 全景 | Gemini 2M / Claude 200k | 60 min |
| **L02** Position Interpolation | Meta 2023 | 60 min |
| **L03** NTK-aware scaling | LocalLlama 社区 | 60 min |
| **L04** YaRN ⭐ | Peng 2023 | 120 min |
| **L05** 3D RoPE / Q-RoPE | 多模态视频 | 60 min |
| **L06** Ring Attention ⭐ | Liu 2023 | 120 min |
| **L07** Striped Attention | — | 60 min |
| **L08** Infini-Attention | Google 2024 | 75 min |
| **L09** FA 长上下文 | block-sparse + window | 60 min |
| **L10** NIAH / RULER 评测 | benchmark | 60 min |
| **L11** 长上下文训练数据 | book/repo packing | 60 min |
| **L12** 长上下文陷阱 | Lost in middle / 注意力稀释 | 60 min |
| **L13** Capstone：Llama-3.2-1B 8k→32k | YaRN + NIAH 验证 | 120 min |

**总学时**: 13 lecture × 平均 75 min + 5h notebook ≈ 14 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机
## Slide 3-6: RoPE 数学 / Ring 并行算法
## Slide 7-10: 推外能力分析
## Slide 11-15: 代码逐行
## Slide 16-20: 实验：NIAH 通过率 / ppl
## Slide 21-25: 思考题 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 | 工业 |
|------|-------------|-----|------|
| Position Interpolation | ✅ | ✅ transformers | — |
| NTK-aware | ✅ | ✅ | — |
| YaRN ⭐ | ✅ | ✅ vllm | — |
| Ring Attention | ✅ naive 单卡 | ✅ ring-flash-attention | (Megatron) |
| Striped | — | ✅ | — |
| Infini-Attention | ✅ 简化 | — | — |
| 3D RoPE | ✅ | — | — |
| NIAH eval | ✅ | — | — |
| RULER eval | — | ✅ NVIDIA RULER | — |

---

## 6. 一致性测试

```python
def test_rope_extrapolation_8k_to_16k():     # ppl 不爆 < 2x
def test_yarn_correctness():                   # YaRN 与论文公式一致
def test_ring_attention_vs_naive():            # 单卡 ring vs naive < 1e-4
def test_niah_pass_rate_32k():                 # ≥ 80%
def test_long_packing_correctness():            # 边界 mask 正确
```

---

## 7. Notebook 结构（13 个）

每个 lecture 一个 ipynb：
1. import + 模型加载
2. RoPE 频率 / Ring 切分可视化
3. minimal 实现
4. mini 实验（NIAH 子集）
5. 库对照
6. 长度外推 ppl 曲线
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (WSL2)
torch>=2.5+cu130
flash-attn>=2.6
ring-flash-attention>=0.1
transformers>=5.0
peft>=0.13  # LoRA for capstone
```

**verify_env.py 三段式**:
- Part A: 基础（flash-attn + ring-flash-attention）
- Part B: GPU + sm_120
- Part C: YaRN scaling smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `lc-rope` | L01-L05: PI + NTK + YaRN + 3D RoPE | 4 |
| `lc-ring` | L06-L07: Ring + Striped | 3 |
| `lc-infini` | L08-L09: Infini + FA long | 2 |
| `lc-eval` | L10-L12: NIAH/RULER + data + 陷阱 | 3 |
| `long-context` | L13: Capstone + README | 3 |

---

## 10. 跨专题衔接

### 上游
- 专题 2 transformer-deep：RoPE / FlashAttention 基础

### 下游
- 专题 7 预训练：Phi-tiny 用 YaRN 扩长
- 专题 8 graduation：评测 Phi-tiny 长上下文能力

### 跨专题对照表预留位
| 方法 | 训长 | 推长 | 通过率 |
|------|------|------|--------|
| Llama-3 原生 | 8k | 8k | baseline |
| + YaRN | 8k | 32k | > 80% |
| + Ring + 32k 训 | 32k | 1M | (需多卡) |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 32k 训练显存 24GB 紧 | 高 | 高 | LoRA + GQA + grad ckpt |
| Ring Attention 真分布式需 ≥4 卡 | 高 | 中 | 单卡 naive 演示 + 多卡云教程 |
| NIAH 通过率不达 | 中 | 中 | 期望管理 80% |
| YaRN 公式实现易错 | 中 | 高 | 严格对照论文 + 单元测 |
| 长上下文数据稀缺 | 中 | 中 | book + repo 拼接，PG-19 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-meta-2023-pi.md
├── 02-locallama-2023-ntk.md
├── 03-peng-2023-yarn.md
├── 04-microsoft-2024-longrope.md
├── 05-liu-2023-ring-attention.md
├── 06-striped-2024.md
├── 07-google-2024-infini-attn.md
├── 08-nope-2023.md
├── 09-niah-2023.md
├── 10-nvidia-2024-ruler.md
├── 11-liu-2023-lost-in-middle.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-long-context.md` 的 6 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L05 RoPE 全套（tag `lc-rope`）
- Phase 3: L06-L07 Ring + Striped（tag `lc-ring`）
- Phase 4: L08-L09 Infini + FA long（tag `lc-infini`）
- Phase 5: L10-L12 评测 + 数据 + 陷阱（tag `lc-eval`）
- Phase 6: L13 Capstone + README（tag `long-context`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
