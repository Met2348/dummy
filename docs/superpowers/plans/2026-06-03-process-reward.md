# Process Reward & Verification Implementation Plan

**Goal**: 完整实现推理 RL "工具箱"学习包（12 lecture + minimal/库两轨 + 数据生成器 + 6 测试 + 12 notebook + GSM8K PRM Capstone）

**Architecture**: 12 lectures = 11 主线 + 1 Capstone。两轨代码：minimal + trl/math-verify/prometheus-eval。

**Tech Stack**: torch 2.13.0 cu130 / transformers 5.x / trl 0.13 / math-verify / sympy / prometheus-eval / networkx

**Design 文档**: `docs/superpowers/specs/2026-06-03-process-reward-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/process-reward/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- trl + math-verify + sympy + prometheus-eval + networkx + matplotlib

### Task 1.3: environment/verify_env.py
- Part A: math-verify import + 1 数学题验证
- Part B: GPU
- Part C: Qwen-0.5B PRM 训练 5-step smoke

### Task 1.4: src/common.py
- GSM8K data loader + answer parser (`####` regex)
- step splitter（用 `\n\n` 或 `Step \d+:`）

### Task 1.5: papers/ 12 个占位 + README

### Commit: `chore: process-reward scaffold`

---

## Phase 2: L1-L3 ORM vs PRM + PRM 训练

### Task 2.1: lectures/01-orm-vs-prm.md (22 slides)
- 概念清理 + 监督粒度对比

### Task 2.2: src/orm_minimal.py
- 整段答案 RM 训练（trl RewardTrainer 适配）

### Task 2.3: lectures/02-lets-verify.md (26 slides)
- OpenAI 800K step label 实验 + PRM800K 数据集

### Task 2.4: lectures/03-prm-training.md (28 slides)
- step 划分 / soft label / loss / 关键工程细节

### Task 2.5: src/prm_minimal.py
- step-level label + cross-entropy loss
- 玩具数据训练

### Task 2.6: src/tests/test_orm_prm.py
- ORM/PRM 玩具准确率 > 60%/70%

### Task 2.7: notebooks/01-03 三个

### Commit + Tag: `prm-base`

---

## Phase 3: L4-L5 Math-Shepherd + PPM

### Task 3.1: lectures/04-math-shepherd.md (28 slides)
- MC rollout 自动生成 PRM 数据完整流程

### Task 3.2: src/math_shepherd_data_gen.py
- MC rollout（每步 N 个 continuation，计算 hard estimation reward）
- 生成 toy PRM jsonl（1k 样本）

### Task 3.3: lectures/05-ppm.md (22 slides)
- preference-based PRM（rStar-Math 用法）

### Task 3.4: src/ppm_minimal.py
- pairwise step preference + BT loss

### Task 3.5: src/tests/test_math_shepherd_data.py
- 100 step 数据合理性验证

### Task 3.6: notebooks/04 + 05

### Commit + Tag: `prm-auto-data`

---

## Phase 4: L6 PRIME 隐式 PRM

### Task 4.1: lectures/06-prime.md (28 slides)
- 隐式 PRM 推导 + 从 outcome 自动学步级

### Task 4.2: src/prime_minimal.py
- 简化版 PRIME（专题 6 会做 full pipeline）

### Task 4.3: src/tests/test_prime_implicit_reward.py
- 隐式 reward 与显式 PRM 相关 > 0.5

### Task 4.4: notebooks/06-prime.ipynb

### Commit + Tag: `prm-prime`

---

## Phase 5: L7 RLVR

### Task 5.1: lectures/07-rlvr.md (22 slides)
- Verifiable Rewards 哲学
- 数学 / 代码 verifier 示例

### Task 5.2: src/rlvr_demo.py
- 数学 verifier（math-verify）
- 代码 verifier（exec sandbox）
- 反 reward-hacking 演示

### Task 5.3: notebooks/07-rlvr.ipynb

### Commit + Tag: `prm-rlvr`

---

## Phase 6: L8-L9 Tree Search + MCTS

### Task 6.1: lectures/08-tree-search.md (24 slides)
- BoN / Beam / ToT 对比
- PRM rerank 流程

### Task 6.2: src/bon_search.py
- Best-of-N 实现 + PRM rerank

### Task 6.3: lectures/09-mcts-llm.md (28 slides)
- MCTS for LLM 完整算法
- Stream of Search + MCTS-DPO

### Task 6.4: src/mcts_llm.py
- 简化 MCTS（UCT 选择 + rollout + backprop）
- 用 networkx 可视化小树

### Task 6.5: src/tests/test_bon_mcts.py
- BoN-32 vs greedy 提升 ≥ 10pp

### Task 6.6: notebooks/08 + 09

### Commit + Tag: `prm-search`

---

## Phase 7: L10-L11 LLM-as-Judge + 陷阱

### Task 7.1: lectures/10-llm-as-judge.md (24 slides)
- G-Eval / Prometheus 2 / 7 个 judge 模型对比

### Task 7.2: src/llm_judge.py
- G-Eval prompt 模板
- Prometheus 2 调用

### Task 7.3: lectures/11-rm-pitfalls.md (22 slides)
- length bias / sycophancy / position bias

### Task 7.4: src/rm_bias_demos.py
- 3 类 bias 反例 demo

### Task 7.5: notebooks/10 + 11

### Commit + Tag: `prm-judge`

---

## Phase 8: L12 Capstone + README

### Task 8.1: lectures/12-capstone-prm-bon.md (28 slides)
- 完整 walkthrough：训 PRM → BoN-32 → 对比 baseline

### Task 8.2: src/capstone_gsm8k_prm/
- 完整 pipeline:
  - 1. Math-Shepherd 数据生成 (5k step-level)
  - 2. PRM 训练（Qwen2.5-0.5B + LoRA）
  - 3. GSM8K-test 100 题：vanilla greedy / majority vote / PRM rerank
  - 4. 输出对照表

### Task 8.3: notebooks/12-capstone-prm-bon.ipynb
- 完整实验展示

### Task 8.4: README.md
- 12 方法横向表 + 决策树（什么时候用 PRM vs RLVR vs LLM-Judge）+ 自测 12 题

### Task 8.5: papers/ 补全

### Commit + Tag: `process-reward`

---

## Phase 9: 收尾验证

### Task 9.1: 全部 notebook 执行
### Task 9.2: 全部测试
### Task 9.3: Capstone PRM rerank > greedy ≥ 10pp
### Task 9.4: verify_env.py PASS

### Final commit: `docs: process-reward README + complete`

---

## 完成验收清单

- [ ] 12 lecture markdown
- [ ] 12 notebook 全跑
- [ ] PRM 玩具准确率 > 70%
- [ ] Math-Shepherd 数据生成器 PASS
- [ ] Capstone PRM rerank 比 greedy 提升 ≥ 10pp
- [ ] BoN-32 vs greedy 提升 ≥ 10pp
- [ ] tag `process-reward`

**预计 git commits**: ~26
**预计实施时长**: 14 hours
