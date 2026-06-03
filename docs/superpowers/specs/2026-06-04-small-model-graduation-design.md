# Small Model + 五部曲毕业 学习专题 — 设计文档

> **承接**: Module 1 (PEFT) + Module 2 (RL/对齐) + Module 3 全 7 专题 完成
> **本专题**: Module 3 第 8 站 ⭐⭐⭐⭐⭐ — 系列收官
> **战略地位**: 整个学习历程（Module 1+2+3 共 116+118=234 方法）的总成果可视化
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

两部分：
1. **小模型时代**：Phi / Llama-3.2 / Qwen3-Small / SmolLM2 / Gemma-3 / BitNet 全图
2. **五部曲毕业作品**：把 Module 1+2+3 全部 ckpt 在同一道题上对照

毕业作品 = 用户整个学习历程的可视化。这不是新算法专题，是 **综合大题**。

### 1.1 为什么需要"小模型+毕业"组合

- 小模型时代（2024-2026）的核心是「数据 > 参数」+「蒸馏 + 量化感知」
- 用户自训的 270M Phi-tiny 本身就是小模型
- 蒸馏到 80M = 完整闭环（造-蒸馏-毕业）
- 毕业作品需要轻量化 model 才能并行加载 5 ckpt

### 1.2 与 Module 2 R1 毕业作品的关系

- Module 2 R1 毕业是 *RL 系列* 收官（5 ckpt: vanilla/LoRA/Adapter/DPO/R1-Zero）
- Module 3 五部曲毕业是 *整个学习历程* 收官（增加"造"的 ckpt）
- 视为"研究生"vs"博士"的两个毕业

---

## 2. 方法清单（14 种）

| # | 方法 | 出处 | 核心 idea |
|---|------|------|----------|
| 1 | **Phi-1 / 1.5 / 2 / 3 / 3.5 / 4** | Microsoft 2023-25 | 教科书数据系列 |
| 2 | **Llama-3.2 1B/3B** | Meta 2024 | 边缘 SOTA |
| 3 | **Qwen3-0.6B/1.7B/4B** | Alibaba 2025 | 多语言小模型 |
| 4 | **SmolLM2** | HF 2024 | 全开源 |
| 5 | **TinyLlama** | StatNLP 2024 | 1B 长训练 |
| 6 | **Gemma-3 4B** | Google 2025 | Google 小 |
| 7 | **Logit Distillation** | Hinton 2015 | KL 散度 |
| 8 | **Hidden State Distillation** | DistilBERT 2019 | MSE / cosine |
| 9 | **Sequence-level Distillation** | 2016 | greedy decode pair |
| 10 | **MiniLLM (reverse KL)** | THU 2023 | 反向 KL 蒸馏 |
| 11 | **BitNet b1.58** | Microsoft 2024 | 1-bit/三值 LLM ⭐ |
| 12 | **Quantization-Aware Pretrain** | — | 训练阶段 QAT |
| 13 | **TinyStories** | Microsoft 2023 | 玩具数据集证明 small 可行 |
| 14 | **五部曲综合** | 自创 | 造-改-用-扩-包 ⭐⭐⭐ |

---

## 3. Lecture 结构（14 篇 = 11 小模型 + 1 蒸馏 + 1 综合 + 1 毕业）

| Lecture | 主题 | 时长 |
|---------|------|------|
| **L01** 小模型时代背景 | 边缘 + 隐私 + 成本 | 45 min |
| **L02** Phi 全系列 | 教科书数据演化 | 75 min |
| **L03** Llama-3.2 1B/3B | 边缘部署 | 60 min |
| **L04** Qwen3-Small | 多语言 | 60 min |
| **L05** SmolLM2 | 全开源 | 45 min |
| **L06** TinyLlama | 1B 长训练 | 45 min |
| **L07** Gemma-3 | Google 小模型 | 45 min |
| **L08** KD 蒸馏 | logit / hidden | 75 min |
| **L09** MiniLLM reverse KL | THU 2023 | 60 min |
| **L10** BitNet b1.58 ⭐ | 1-bit LLM | 90 min |
| **L11** QAT 量化感知预训 | 训练阶段 QAT | 60 min |
| **L12** Capstone-1：270M → 80M 蒸馏 | logit+hidden | 120 min |
| **L13** 五部曲综合理论 ⭐⭐⭐ | 造-改-用-扩-包统一公式 | 90 min |
| **L14** Capstone-2：五部曲毕业作品 ⭐⭐⭐⭐⭐ | 同 GSM8K × 5 路径对照 | 180 min |

**总学时**: 14 lecture × 平均 70 min + 4h notebook ≈ 12 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（小模型场景 / 蒸馏目的）
## Slide 3-6: 方法核心
## Slide 7-10: 实战案例（Phi-4 / Llama-3.2 真案例）
## Slide 11-15: 代码逐行（仅 capstone L12-L14）
## Slide 16-20: 实验：student vs teacher / quant vs FP16
## Slide 21-25: 思考题 + 下节预告
```

**毕业 lecture L14 特殊结构（32 slides）**：
- Part I（8 slides）：五线回顾
- Part II（12 slides）：统一公式 `p(y|x; θ_data, θ_arch, θ_weight, φ)`
- Part III（8 slides）：工程选型决策树（4 个真实场景）
- Part IV（4 slides）：历史观 + 下一程

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 | 工业 |
|------|-------------|-----|------|
| Phi/Llama/Qwen3-Small 加载 | — | ✅ transformers | — |
| Logit distill | ✅ KL | — | — |
| Hidden distill | ✅ MSE / cosine | — | — |
| MiniLLM reverse KL | ✅ | — | — |
| BitNet forward | ✅ 三值量化 | — | (官方 repo) |
| QAT pretrain | ✅ 教学版 | — | — |
| 270M → 80M 蒸馏 | ✅ | — | — |
| 五部曲毕业 | ✅⭐ 加载 5 ckpt | — | — |

---

## 6. 一致性测试

```python
def test_logit_distill_loss():            # 数学公式正确
def test_hidden_distill_dim_match():       # 维度对齐 layer
def test_bitnet_forward():                  # 三值量化 forward 正确
def test_qat_train_loss_close():           # QAT vs FP train loss < 5% diff
def test_distill_student_ppl():            # student < teacher × 1.3
def test_capstone_load_5_ckpts():           # 5 个 ckpt 全加载成功
def test_capstone_five_method_outputs():    # 5 个 response 长度不同
def test_capstone_export_for_notebook():    # 导出 dict 结构正确
```

---

## 7. Notebook 结构（14 个）

特别说明：
- L13-L14 notebook 是毕业**核心交付物**
- 包含 5 ckpt 加载 + GSM8K 题 + 5 个 response + 性能对照 + 雷达图

---

## 8. 环境配置

```
# requirements.txt (WSL2)
torch>=2.5+cu130
transformers>=5.0
peft>=0.13          # 加载 LoRA ckpt
bitsandbytes>=0.43  # 4bit 加载多个 model
accelerate>=1.0
matplotlib seaborn  # 雷达图
```

**verify_env.py 三段式**:
- Part A: 基础（transformers + peft + bnb）
- Part B: GPU + sm_120 + 8GB 空闲
- Part C: 加载 Phi-3.5 mini 4bit smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `small-models` | L01-L07: 小模型 6 系列 | 5 |
| `distill` | L08-L09: KD + MiniLLM | 3 |
| `bitnet-qat` | L10-L11: BitNet + QAT | 2 |
| `distill-capstone` | L12: 270M→80M | 2 |
| `five-unification` | L13: 五部曲理论 | 2 |
| `造改-graduation` | L14: 毕业作品 ⭐⭐⭐⭐⭐ | 3 |

---

## 10. 跨专题衔接

### 上游
- Module 1 PEFT 全部 ckpt（vanilla GPT-2 / LoRA / Adapter / Prompt）
- Module 2 RL 全部 ckpt（DPO / R1-Zero / VLM-R1）
- Module 3 专题 7 ckpt（自训 270M Phi-tiny）

### 下游
- 无（系列毕业）
- 可选：Module 4-6（推理 / 扩 / 包）继续深造

### 五部曲毕业作品具体设计

**同一道 GSM8K 题** + **5 个 model 演化路径**：

| # | 路径 | ckpt 来源 | 改了什么 |
|---|------|---------|---------|
| 1 | **Vanilla GPT-2** | 公开 | (baseline) |
| 2 | **LoRA 微调** | Module 1 L01 ckpt | weight (low-rank) |
| 3 | **DPO 对齐** | Module 2 dpo-family ckpt | distribution shape |
| 4 | **R1-Zero 推理** | Module 2 reasoning-r1 ckpt | trajectory |
| 5 | **自训 Phi-tiny** ⭐ | Module 3 专题 7 ckpt | 整个 model (造) |

**对照可视化**：
- 5 种 response 文本对照
- 5 种 latency / 显存对照
- 雷达图：格式 / 准确 / 推理深度 / 自检 / 响应速度
- 经验观察：从 0 训 270M 不一定胜过 GPT-2 + LoRA，但展示了「能造的能力」

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 5 ckpt 全加载显存 ~40GB | 高 | 高 | 4bit 加载 + 顺序加载（不并发）|
| 自训 ckpt 性能弱 | 高 | 中 | 期望管理「展示能力 != 性能」 |
| 用户 ckpt 缺失 | 中 | 高 | 提供 fallback 下载 |
| BitNet 实现复杂 | 中 | 低 | 教学版三值量化 |
| MiniLLM reverse KL 不收敛 | 中 | 中 | 严格按论文 + 充足训练 step |
| 五部曲理论 L13 抽象 | 中 | 低 | 配 32 slides 详细推导 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-phi-series-all.md          # Phi-1 → Phi-4 系列
├── 02-llama-3.2-2024.md
├── 03-qwen3-2025.md
├── 04-smollm2-2024.md
├── 05-tinyllama-2024.md
├── 06-gemma-3-2025.md
├── 07-hinton-2015-distillation.md
├── 08-distilbert-2019.md
├── 09-tinystories-2023.md
├── 10-minillm-thu-2023.md
├── 11-bitnet-b1.58-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-small-model-graduation.md` 的 6 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L07 小模型 6 系列（tag `small-models`）
- Phase 3: L08-L09 蒸馏（tag `distill`）
- Phase 4: L10-L11 BitNet + QAT（tag `bitnet-qat`）
- Phase 5: L12 Capstone-1（tag `distill-capstone`）
- Phase 6: L13-L14 五部曲毕业（tag `five-unification` → `造改-graduation`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
- **特别说明**: 本专题完结后整个 234 方法学习系列毕业
