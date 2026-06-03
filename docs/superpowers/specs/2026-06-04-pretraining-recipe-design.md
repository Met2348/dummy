# Pretraining Recipe 学习专题 — 设计文档

> **承接**: 专题 1 数据 + 专题 2 架构 + 专题 6 infra（三件套 ready）
> **本专题**: Module 3 第 7 站 ⭐⭐⭐ — 系列高峰
> **战略地位**: 从 0 训完整 270M Phi-tiny，把 Module 3 前面所有内容串成 capstone
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

预训练的"完整 recipe" = 数据 + 架构 + infra + warmup/cosine + optimizer + 监控 + 评测 + ckpt 管理。本专题用 Phi-style 270M 模型作 capstone，跑完整 5B-token 预训练，让用户体验真训挑战。

### 1.1 为什么这是高峰

- 前 6 专题分别在解决 *局部问题*；本专题是 *整合*
- 真预训练 20h+ ≠ 玩具 demo 1h
- loss spike / lr 调度 / data annealing / mid-training 等 *真训才遇见* 的问题
- 失败概率 30%+（这是预训练的真实样子）

### 1.2 本专题的"教学策略"

- 单卡 5090 可跑 ~20h（紧 + 慢）
- 或推荐租云 4×A100 ~4h（~$40，省时）
- 提供预训 ckpt 下载 fallback（卡住或失败的学员可继续后续 lecture）

---

## 2. 方法清单（18 种）

| # | 方法 | 出处 | 核心 idea |
|---|------|------|----------|
| 1 | **AdamW** | Loshchilov 2017 | weight decay 解耦 |
| 2 | **Lion** | Chen 2023 | sign-based optimizer |
| 3 | **Sophia** | Stanford 2023 | Hessian-aware |
| 4 | **Warmup-Stable-Decay (WSD)** | DeepSeek | 三段 lr schedule ⭐ |
| 5 | **Linear Warmup** | — | 标准 |
| 6 | **Cosine Decay** | — | 标准 |
| 7 | **Gradient Clipping** | — | 防爆 |
| 8 | **Loss Spike Handler** | — | rollback / skip ⭐ |
| 9 | **μP Init** | Yang 2022 | 超参 transfer |
| 10 | **Xavier / He Init** | 经典 | 各种 init |
| 11 | **Dropout / WD strategy** | — | 早期 / 后期 |
| 12 | **StreamingDataset** | Mosaic 2023 | 流式多文件 ⭐ |
| 13 | **WebDataset** | — | tar 格式流式 |
| 14 | **EMA (Exponential Moving Avg)** | — | model averaging |
| 15 | **Model Soup** | 2022 | ckpt averaging |
| 16 | **Phi Recipe** | Microsoft 2023-25 | 教科书 + 合成数据 |
| 17 | **Llama-3 Recipe** | Meta 2024 | 15T + long ctx + annealing |
| 18 | **Data Annealing** | 2024 | 末段切高质量 ⭐ |

---

## 3. Lecture 结构（16 篇 = 14 主线 + 2 capstone）

| Lecture | 主题 | 时长 |
|---------|------|------|
| **L01** 现代预训练全流程 | data → tokenizer → model → train → eval | 60 min |
| **L02** lr schedule | warmup / cosine / WSD ⭐ | 75 min |
| **L03** optimizer 选型 | AdamW / Lion / Sophia | 75 min |
| **L04** Gradient clip + loss spike | rollback / skip / restart | 75 min |
| **L05** 初始化策略 | Xavier / He / μP init | 60 min |
| **L06** Dropout / WD strategy | 早期 / 后期 | 45 min |
| **L07** 数据加载 | StreamingDataset / WebDataset | 75 min |
| **L08** Loss 监控 | wandb / tensorboard / 早期警报 | 60 min |
| **L09** 训练时评测 | val loss / downstream tasks | 60 min |
| **L10** Checkpoint 管理 | resume / EMA / averaging | 75 min |
| **L11** Phi 系列 recipe | 教科书 + 合成数据 | 90 min |
| **L12** Llama-3 recipe | 15T + 长上下文 + annealing | 75 min |
| **L13** Data annealing | 末段切高质量 ⭐ | 60 min |
| **L14** mid-training | continued pretraining | 60 min |
| **L15** Capstone：从 0 训 270M Phi-tiny ⭐⭐⭐ | 完整 20h 训练 | 180 min |
| **L16** Capstone 评测 | MMLU / GSM8K / HumanEval 子集 | 75 min |

**总学时**: 16 lecture × 平均 75 min + 8h notebook ≈ 20 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（预训练痛点）
## Slide 3-6: 算法 / 配方
## Slide 7-10: 实战数据（Phi/Llama 真训案例）
## Slide 11-15: 代码逐行
## Slide 16-20: 实验：调对 vs 调错对照
## Slide 21-25: 思考题 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 | 工业 |
|------|-------------|-----|------|
| AdamW | (torch 内置) | ✅ torch.optim | — |
| Lion | ✅ | ✅ lion-pytorch | — |
| Sophia | ✅ | ✅ sophia | — |
| WSD scheduler | ✅ | — | — |
| Loss spike handler | ✅ | — | (DeepSpeed) |
| μP init | ✅ | ✅ mup | — |
| StreamingDataset | ✅ 玩具 | ✅ mosaic streaming | — |
| EMA | ✅ | — | — |
| Model averaging | ✅ | — | — |
| eval harness | — | ✅ lm-eval-harness | — |
| Phi-tiny 完整训练 | ✅ 全配方 | (megatron / mosaic 可选) | — |

---

## 6. 一致性测试

```python
def test_lion_vs_adamw_step():            # 玩具任务 Lion 收敛
def test_wsd_schedule_correctness():       # 三段 lr 数值
def test_loss_spike_rollback():           # 注入 spike,自动 rollback
def test_mup_lr_transfer():                # 64 → 256 hidden lr 一致
def test_streaming_data_loader():          # 多文件迭代正确
def test_ema_correctness():                # 数学公式正确
def test_resume_continuity():              # resume 后 loss 连续
def test_eval_harness_smoke():            # MMLU 5 题 PASS
```

---

## 7. Notebook 结构（16 个）

每个 lecture 一个 ipynb：
1. import + 模型 + 数据
2. 算法 / schedule 可视化
3. minimal 实现
4. mini 训练（500 step）
5. 真实预训练 log 对照
6. 思考题

---

## 8. 环境配置

```
# requirements.txt (WSL2 + 推荐 ≥1 卡)
torch>=2.5+cu130
deepspeed>=0.15
megatron-core>=0.9
streaming>=0.7      # Mosaic
lm-eval-harness>=0.4
wandb
lion-pytorch
sophia
mup>=1.0
```

**verify_env.py 三段式**:
- Part A: 基础（torch + deepspeed + mosaic streaming）
- Part B: GPU + sm_120
- Part C: 5 step 训练 smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `pretrain-lr` | L01-L02: lr schedule | 2 |
| `pretrain-optim` | L03: optimizer | 2 |
| `pretrain-stability` | L04-L06: spike + init + reg | 3 |
| `pretrain-data` | L07: 数据加载 | 2 |
| `pretrain-monitor` | L08-L10: 监控 + eval + ckpt | 3 |
| `pretrain-recipes` | L11-L14: Phi/Llama recipe + annealing + mid | 4 |
| `pretraining-recipe` | L15-L16: Capstone + 评测 + README | 5 |

---

## 10. 跨专题衔接

### 上游
- 专题 1: 数据 + tokenizer
- 专题 2: 架构（Phi-tiny 用 RoPE+GQA+SwiGLU+RMSNorm）
- 专题 6: infra（FSDP 训练）

### 下游
- 专题 8: 把 270M Phi-tiny 蒸馏 + 当 graduation 五部曲的 "造" 路径

### 跨专题对照表预留位
| 模型 | 参数 | 数据 | 训练 | 性能 |
|------|------|------|------|------|
| 自训 Phi-tiny | 270M | 5B token | 20h × 5090 | val ppl<25 |
| Phi-4 14B | 14B | 9.8T | 21d × A100 | MMLU 84 |
| Llama-3 8B | 8B | 15T | 1.3M GPU-h | MMLU 66 |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 270M × 5B 算力 1.4 PFLOPS·h | 高 | 中 | 5090 24h 紧；推荐云 4×A100 ~$40 |
| 真训失败概率 30%+ | 高 | 中 | 提供 ckpt 下载 fallback |
| Capstone 评测 underperform | 高 | 低 | 期望管理（270M 类似 GPT-2-small）|
| loss spike 训挂 | 中 | 高 | 自动 rollback + 严格梯度 clip |
| StreamingDataset Windows 兼容 | 高 | 中 | 强制 WSL2 + 提供 known-good |
| μP transfer 实测失败 | 中 | 低 | L05 提供小规模验证脚本 |
| eval-harness 慢 | 中 | 低 | 子集 (200 题) 评测 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-loshchilov-2017-adamw.md
├── 02-chen-2023-lion.md
├── 03-liu-2023-sophia.md
├── 04-deepseek-wsd.md
├── 05-bingrong-2024-loss-spike.md
├── 06-yang-2022-mup.md
├── 07-mosaic-streaming-2023.md
├── 08-wortsman-2022-model-soup.md
├── 09-phi-1-2023.md
├── 10-phi-2-2023.md
├── 11-phi-3-2024.md
├── 12-phi-4-2024.md
├── 13-llama-3-2024.md
├── 14-llama-3.3-2024.md
├── 15-data-annealing-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-pretraining-recipe.md` 的 7 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L02 lr schedule（tag `pretrain-lr`）
- Phase 3: L03 optimizer（tag `pretrain-optim`）
- Phase 4: L04-L06 稳定性 + init（tag `pretrain-stability`）
- Phase 5: L07 数据加载（tag `pretrain-data`）
- Phase 6: L08-L10 监控 + eval + ckpt（tag `pretrain-monitor`）
- Phase 7: L11-L14 recipes + annealing + mid（tag `pretrain-recipes`）
- Phase 8: L15-L16 Capstone + README（tag `pretraining-recipe`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
