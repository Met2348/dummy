# Process Reward & Verification 学习专题 — 设计文档

> **承接**: dpo-family (偏好优化主线)
> **本专题**: 推理 RL 的"工具箱" — PRM + verifier + MCTS + BoN
> **战略地位**: 专题 5 R1 时代的关键前置
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

Process Reward Model (PRM) 是从"整段答案打分"（ORM）到"每步打分"（PRM）的关键演化。本专题教会用户：
1. **怎么训 PRM**（手动标注 vs 自动生成 Math-Shepherd 风格）
2. **怎么做 verifier**（程序 / 规则 / LLM-as-Judge）
3. **怎么用 PRM 组合搜索**（BoN / Beam / MCTS）
4. **2025 SOTA**: PRIME (隐式 PRM)、Skywork-Reward V2、Self-Taught Evaluator

### 1.1 为什么 PRM 是推理 RL 的关键
- **ORM 限制**：只看最终答案对错，无法引导中间推理
- **PRM 优势**：每步 reward signal，更密集，更适合长 CoT
- **R1 的选择**：DeepSeek-R1 用 rule-based + ORM（绕过 PRM 训练难度），但 rStar-Math / PRIME 走 PRM 路线

### 1.2 本专题与专题 5 R1 的衔接
- 专题 5 学完后，可以基于本专题的 PRM/verifier 工具做"R1+PRM"对照实验
- 专题 6 SOTA 中的 PRIME 是本专题的算法升级

---

## 2. 方法清单（12 种）

| # | 方法 | 年份 | 论文 | 核心 idea |
|---|------|------|------|----------|
| 1 | **ORM vs PRM 对比** | — | — | 概念清理 |
| 2 | **Let's Verify Step by Step** | 2023.05 | Lightman et al., OpenAI | 第一个 large-scale PRM |
| 3 | **PRM 训练流程** | — | — | step 划分 + soft label |
| 4 | **Math-Shepherd** | 2024 | Wang et al. | 自动 PRM 数据生成（MC rollout）|
| 5 | **PPM** | 2025.01 | rStar-Math | preference-based PRM |
| 6 | **PRIME** | 2025.02 | THU+Microsoft | 隐式 PRM，从 outcome 自学 |
| 7 | **RLVR** | 2024-2025 | — | Verifiable Rewards，规则代替 RM |
| 8 | **Tree Search + Verifier** | — | — | BoN / Beam / ToT |
| 9 | **MCTS for LLM** | 2024-2025 | Stream of Search / MCTS-DPO | 树搜索推理 |
| 10 | **LLM-as-Judge 系统化** | 2024 | G-Eval / Prometheus 2 | 用 LLM 当 RM |
| 11 | **JudgeBench** | 2024 | — | LLM-Judge 评估基准 |
| 12 | **RM 陷阱合集** | — | — | length / sycophancy / position bias |

---

## 3. Lecture 结构（12 篇 = 11 主线 + 1 capstone）

| Lecture | 主题 | 时长 |
|---------|------|------|
| **L1** ORM vs PRM | 概念 + 监督粒度差异 | 60 min |
| **L2** Let's Verify Step by Step | OpenAI 800K step label 实验 | 60 min |
| **L3** PRM 训练实战 | step 划分 + soft label + loss | 90 min |
| **L4** Math-Shepherd | MC rollout 自动生成 PRM 数据 | 90 min |
| **L5** PPM (rStar-Math) | preference-based PRM | 60 min |
| **L6** PRIME (隐式 PRM) | 从 outcome 自动学步级 | 90 min |
| **L7** RLVR | Verifiable Rewards (数学/代码) | 60 min |
| **L8** Tree Search + Verifier | BoN / Beam / ToT | 90 min |
| **L9** MCTS for LLM | Stream of Search + MCTS-DPO | 90 min |
| **L10** LLM-as-Judge | G-Eval / Prometheus 2 | 60 min |
| **L11** RM 陷阱合集 | length bias 等 | 60 min |
| **L12** Capstone：GSM8K PRM + BoN | 训 PRM + Best-of-32 推理 | 120 min |

**总学时**: 11×72 + 120 = 912 min ≈ 15 hours

---

## 4. Lecture 模板（每篇 22-26 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 这个方法在 RM 演化谱系中的位置
## Slide 3-5: 数据准备（关键：PRM 数据怎么来）
## Slide 6-10: 训练流程（loss + 监督信号 + 工程细节）
## Slide 11-14: 代码逐行（minimal 实现）
## Slide 15-17: 推理时怎么用（PRM rerank / MCTS guided）
## Slide 18-20: 实验观察 / 论文 claim
## Slide 21-23: 陷阱与警示
## Slide 24-25: 下节预告
```

---

## 5. 代码两轨 + 数据轨

| 方法 | minimal | trl/库 | 数据生成 |
|------|---------|--------|---------|
| ORM | ✅ 手写 | trl RewardTrainer | — |
| PRM 训练 | ✅ 手写 | trl 适配 | — |
| Math-Shepherd | ✅ 数据生成器 | — | ⭐ 核心 |
| PRIME | ✅ 隐式 PRM 推导 | — | — |
| RLVR | ✅ 数学/代码 verifier | math-verify 库 | — |
| BoN | ✅ 手写 | — | — |
| MCTS | ✅ 简化树搜索 | — | — |
| LLM-as-Judge | ✅ 模板 | prometheus-eval | — |

**目录约定**:
- `orm_minimal.py` / `prm_minimal.py` — ORM 和 PRM 训练
- `math_shepherd_data_gen.py` — 自动 PRM 数据生成（⭐ 核心）
- `prime_minimal.py` — PRIME 隐式 PRM
- `rlvr_demo.py` — 程序 reward 演示
- `bon_search.py` / `mcts_llm.py` — 推理时组合
- `llm_judge.py` — Judge prompt 模板

---

## 6. 一致性测试

```python
def test_orm_acc():              # ORM 玩具数据准确率 > 60%
def test_prm_step_acc():         # PRM 步骤级准确率 > 70%
def test_math_shepherd_data():   # 生成 100 step 数据合理
def test_prime_implicit_reward():# 隐式 reward 与显式 PRM 相关 > 0.5
def test_bon_improvement():      # BoN-32 vs greedy 提升 ≥ 10pp
def test_rlvr_safety():          # RLVR 不会被 hack
```

---

## 7. Notebook 结构（12 个）

每个 lecture 一个 ipynb：
1. PRM 数据示例（带 step label）
2. 训练 loop step-by-step
3. 推理时如何用 PRM（rerank 演示）
4. 与 ORM 对照实验
5. MCTS 可视化（小树）
6. 思考题

---

## 8. 环境配置

```
# requirements.txt (Windows native)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
math-verify              # RLVR 数学验证
sympy                    # 数学符号
prometheus-eval          # LLM-as-Judge
networkx                 # MCTS 可视化
matplotlib, seaborn
```

**verify_env.py**:
- Part A: math-verify import + 一个数学题验证
- Part B: GPU
- Part C: Qwen-0.5B + 玩具 PRM 训练 smoke test

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `prm-base` | L1-L3: ORM vs PRM + 训练流程 | 4 |
| `prm-auto-data` | L4-L5: Math-Shepherd + PPM | 4 |
| `prm-prime` | L6: PRIME 隐式 PRM | 2 |
| `prm-rlvr` | L7: RLVR + verifier | 2 |
| `prm-search` | L8-L9: BoN + MCTS | 4 |
| `prm-judge` | L10-L11: LLM-as-Judge + 陷阱 | 3 |
| `process-reward` | L12: Capstone GSM8K PRM + BoN | 3 |

---

## 10. 跨专题衔接

### 与专题 5 R1 的关键铺垫
- 专题 5 用 rule-based RLVR + ORM，但学完本专题后用户能扩展到 PRM
- L7 RLVR 直接是 R1-Zero 的 reward 设计基础

### 与专题 6 SOTA 的衔接
- L6 PRIME 在专题 6 会展开 full pipeline
- Skywork-Reward V2 / Self-Taught Evaluator 在专题 6 单独讲

### 与专题 3 DPO 的衔接
- L9 MCTS-DPO 是 DPO 在树搜索上的扩展
- L8 BoN 用 RM rerank，与 DPO 的 preference 视角对应

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| PRM 数据生成慢（MC rollout）| 高 | 高 | 提供预生成 10k jsonl 下载 |
| GSM8K parse `####` 错 | 中 | 中 | L3 专讲 regex，严格测试 |
| PRM 玩具准确率有限 | 中 | 中 | 期望管理 70% |
| MCTS 实现复杂 | 高 | 中 | 用 networkx 可视化简化 |
| LLM-as-Judge 不稳定 | 中 | 低 | 用 Prometheus 2 + 多采样平均 |
| RLVR 在非数学/代码任务不适用 | 高 | 低 | 限定本专题只用数学/代码示例 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-lets-verify-2023.md           # Lightman et al., OpenAI
├── 02-math-shepherd-2024.md
├── 03-rstar-math-ppm-2025.md
├── 04-prime-2025.md
├── 05-rlvr-2024.md                  # 综合 review
├── 06-tot-2023.md                   # Tree of Thoughts
├── 07-stream-of-search-2024.md
├── 08-mcts-dpo-2024.md
├── 09-g-eval-2023.md
├── 10-prometheus-2-2024.md
├── 11-judgebench-2024.md
├── 12-rm-bias-2024.md               # length / sycophancy bias review
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-03-process-reward.md` 的 7 个 Phase 推进。Capstone (L12) 用 Qwen2.5-0.5B + GSM8K：训 PRM (5k step-level 数据) + Best-of-32 rerank + 与 majority vote / greedy 对照，单卡 5090 总 ≤ 4h。
