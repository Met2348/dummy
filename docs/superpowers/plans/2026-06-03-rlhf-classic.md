# RLHF Classic Implementation Plan

**Goal**: 完整实现 RLHF 经典学习包（12 lecture + 三段管线 minimal/trl + 6 测试 + 12 notebook + Capstone TL;DR）

**Architecture**: 12 lectures = 11 主线 + 1 capstone。两轨代码：minimal + trl。复用 PEFT cu130 Windows native 环境。

**Tech Stack**: torch 2.13.0 cu130 / transformers 5.x / trl 0.13 / peft 0.19 / datasets

**Design 文档**: `docs/superpowers/specs/2026-06-03-rlhf-classic-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/rlhf-classic/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- trl + datasets + accelerate + scipy + wandb + matplotlib

### Task 1.3: environment/verify_env.py
- Part A: trl 各 Trainer init（SFT/Reward/PPO）
- Part B: GPU
- Part C: 三段 5-step smoke test

### Task 1.4: src/common.py
- 复用 rl-foundations 的 PPO helpers
- 新增：preference data loader / dialog format / Anthropic-HH 1k 子集脚本

### Task 1.5: papers/ 9 个占位 + README

### Commit: `chore: rlhf-classic scaffold`

---

## Phase 2: L1-L3 三段式定义 + SFT + RM

### Task 2.1: lectures/01-instructgpt.md (28 slides)
- InstructGPT 三段式全景 + 历史背景 + 数据规模

### Task 2.2: lectures/02-sft.md (22 slides)
- SFT 数据格式 / loss / 与预训练区别

### Task 2.3: src/sft_minimal.py + sft_trl.py
- 手写 SFT loop + trl SFTTrainer 对照
- 在 GPT-2 + Alpaca-1k 子集上

### Task 2.4: lectures/03-reward-model.md (28 slides)
- BT loss 完整推导（DPO 推导的前置）
- pairwise data 格式 / value head

### Task 2.5: src/rm_minimal.py + rm_trl.py
- 手写 BT-loss 训练
- trl RewardTrainer 对照
- GPT-2 + Anthropic-HH 1k

### Task 2.6: src/tests/test_sft_consistency.py + test_rm_bt_loss.py
- SFT loss < 1e-6
- BT loss 数值正确
- RM 准确率 > 60%

### Task 2.7: notebooks/01-03 三段对应

### Commit + Tag: `rlhf-sft-rm`

---

## Phase 3: L4-L5 PPO for LLM 深化 + 工程细节

### Task 3.1: lectures/04-ppo-for-llm-deep.md (28 slides)
- token-level reward / KL ref penalty / 4 model 协同 / value head 加在 LM

### Task 3.2: src/ppo_llm_minimal.py
- 完整手写 4 model PPO（actor/critic/ref/RM）
- KL adaptive controller

### Task 3.3: src/ppo_llm_trl.py
- trl PPOTrainer 完整 demo

### Task 3.4: lectures/05-rlhf-engineering.md (26 slides)
- rollout batch / KL adaptive / advantage norm / reward whitening

### Task 3.5: src/rlhf_engineering_tricks.py
- 5 个工程 trick 单独消融

### Task 3.6: src/tests/test_ppo_pipeline.py
- 三段都跑通 + 中间产物保存
- KL < 10 验证

### Task 3.7: notebooks/04 + 05

### Commit + Tag: `rlhf-ppo-llm`

---

## Phase 4: L6-L8 RLAIF + LLaMA-2 + Sparrow

### Task 4.1: lectures/06-rlaif-cai.md (24 slides)
- Constitutional AI principles / 自我批评 / RLAIF 流程

### Task 4.2: src/cai_demo.py
- 简化版：用 GPT-2 + 一条 principle 演示自我批评

### Task 4.3: lectures/07-llama2-rlhf.md (22 slides)
- rejection sampling / iterative RLHF / Meta 工程细节

### Task 4.4: src/rejection_sampling.py
- BoN rejection sampling 实现

### Task 4.5: lectures/08-sparrow.md (20 slides)
- Sparrow 规则约束 + 检索 RLHF

### Task 4.6: notebooks/06 + 07 + 08

### Commit + Tag: `rlhf-rlaif`

---

## Phase 5: L9-L11 陷阱 + 多目标

### Task 5.1: lectures/09-reward-hacking.md (24 slides)
- Gao et al. 2022 over-optimization / 长度 hack 复现实验

### Task 5.2: src/reward_hacking_demo.py
- 用长度 reward 训 GPT-2 → 学到超长文本 hack
- Gao curve 复现

### Task 5.3: lectures/10-rlhf-pitfalls.md (22 slides)
- length bias / sycophancy / 对齐税 / mode collapse

### Task 5.4: lectures/11-multiobj-rlhf.md (22 slides)
- MaxMin-RLHF / Pareto frontier / 双 RM 加权

### Task 5.5: src/multiobj_rlhf.py
- helpful + harmless 双 RM 加权训练

### Task 5.6: src/tests/test_reward_hacking_visible.py
- 长度 reward 后生成长度增长 > 2x

### Task 5.7: notebooks/09 + 10 + 11

### Commit + Tag: `rlhf-pitfalls`

---

## Phase 6: L12 Capstone + README

### Task 6.1: lectures/12-capstone-tldr-rlhf.md (28 slides)
- 完整 SFT → RM → PPO 三段管线 walkthrough

### Task 6.2: src/capstone_tldr/
- 完整三段管线代码（数据 + 模型 + 训练 + 评估）
- GPT-2-medium + summarize_from_feedback 1k 子集

### Task 6.3: notebooks/12-capstone-tldr-rlhf.ipynb
- 完整跑通 + 中间产物 + 最终 spot-check 5 样本

### Task 6.4: README.md
- 12 方法横向表 + 学习路径 + 跨专题衔接表 + 自测 12 题

### Task 6.5: papers/ 补全

### Commit + Tag: `rlhf-classic`

---

## Phase 7: 收尾验证

### Task 7.1: 全部 notebook 执行
### Task 7.2: 全部测试
### Task 7.3: Capstone RM 准确率 > 60% + PPO 后 RM-score 提升 > 0.3
### Task 7.4: verify_env.py PASS

### Final commit: `docs: rlhf-classic README + complete`

---

## 完成验收清单

- [ ] 12 lecture markdown
- [ ] 12 notebook 全跑
- [ ] minimal vs trl 三段一致性 PASS
- [ ] Capstone TL;DR 跑通，RM-score 提升 ≥ 0.3
- [ ] verify_env.py 全 PASS
- [ ] tag `rlhf-classic`

**预计 git commits**: ~25
**预计实施时长**: 15 hours
