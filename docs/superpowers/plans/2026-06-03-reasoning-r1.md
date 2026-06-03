# Reasoning R1 Implementation Plan ⭐⭐⭐⭐⭐

**Goal**: 完整实现 R1 时代学习包（15 lecture + minimal/trl/verl 三轨 + 双轨 Capstone（GPT-2-M + Qwen-1.5B 复现 R1-Zero））

**Architecture**: 15 lectures = 1 WSL2 setup + 12 主线 + 2 双轨 Capstone。三轨代码：minimal + trl + **verl**（DAPO/Open-R1 公认基础设施）。

**⚠️ 环境切换**: 本专题第一个 lecture (L0) 切换到 WSL2。verl + Ray + Megatron + vllm 在 Windows native 装不上。

**Tech Stack**: WSL2 Ubuntu 22.04 / torch cu130 / transformers 5.x / trl 0.13 / **verl ≥0.4** / **vllm ≥0.7** / Ray / peft / bitsandbytes / math-verify

**Design 文档**: `docs/superpowers/specs/2026-06-03-reasoning-r1-design.md`

---

## Phase 1: L0 WSL2 + verl + vllm 环境

### Task 1.1: lectures/00-wsl2-setup.md (28 slides)
- Windows Terminal + WSL2 Ubuntu 22.04 配置
- CUDA 13.0 + cuDNN
- Python 3.11 venv
- verl + vllm 安装步骤（含已知坑）
- 从 Windows 文件系统访问 `/mnt/c/Workspace/dummy`

### Task 1.2: WSL2 目录骨架
- WSL2 中创建 `learning/reasoning-r1/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.3: environment/requirements.txt
- verl 0.4+ / vllm 0.7+ / trl 0.13+ / ray 2.30+ / torch cu130 / bitsandbytes / math-verify

### Task 1.4: environment/verify_env.py
- Part A: verl + vllm import
- Part B: vllm 单卡 inference smoke
- Part C: verl GRPO 5-step smoke + Ray cluster init

### Task 1.5: known-good Dockerfile 备份（如本地装不上）

### Task 1.6: src/common.py
- rollout helpers / reward function 接口 / format check / answer parser

### Task 1.7: papers/ 12 个占位 + README

### Commit + Tag: `r1-wsl2-setup`

---

## Phase 2: L1-L2 o1 + GRPO 推导

### Task 2.1: lectures/01-o1-paradigm.md (24 slides)
- 推理时 RL scaling 范式定义
- o1/o3 历史 + Test-time compute 概念引入

### Task 2.2: lectures/02-grpo-derivation.md (28 slides)
- GRPO 完整数学推导
- 与 PPO 对比（去 critic / group baseline / 优势归一化）

### Task 2.3: src/grpo_minimal.py
- 手写 GRPO（核心算法）
- 关键：group advantage 计算 + KL constraint

### Task 2.4: src/grpo_trl.py
- trl GRPOTrainer demo

### Task 2.5: src/grpo_verl.py
- verl 配置文件 + 启动脚本
- 在 Qwen2.5-0.5B + 玩具数据上 5-step smoke

### Task 2.6: src/tests/test_grpo_consistency.py
- minimal vs trl loss < 1e-6

### Task 2.7: notebooks/01 + 02

### Commit + Tag: `r1-grpo-core`

---

## Phase 3: L3-L5 DeepSeek-R1 / R1-Zero / Kimi k1.5

### Task 3.1: lectures/03-r1-zero.md (28 slides)
- R1-Zero 训练算法
- aha moment 涌现机制
- 关键观察：wait / recheck / let me reconsider

### Task 3.2: lectures/04-r1.md (26 slides)
- cold-start + 4 阶段训练
- 与 R1-Zero 区别

### Task 3.3: lectures/05-kimi-k1.5.md (24 slides)
- long context RL + 多模态 RL 联合
- Moonshot 工程细节

### Task 3.4: src/r1_zero_algo.py
- R1-Zero 算法核心（GRPO + rule-based reward + cold-start ✗）

### Task 3.5: src/rewards/format_reward.py + accuracy_reward.py + combined_reward.py
- format: `<think>...</think><answer>...</answer>` regex
- accuracy: GSM8K parse `####` / Countdown 计算 ==

### Task 3.6: src/tests/test_format_accuracy_reward.py
- 严格单元测试 reward function

### Task 3.7: notebooks/03 + 04 + 05

### Commit + Tag: `r1-deepseek`

---

## Phase 4: L6-L9 算法变体 RLOO / ReMax / VinePPO / REINFORCE++

### Task 4.1: lectures/06-rloo.md (22 slides)
- Leave-One-Out baseline 数学

### Task 4.2: src/rloo_minimal.py + rloo_trl.py + rloo_verl.py

### Task 4.3: lectures/07-remax.md (22 slides)
- 去 critic 简化

### Task 4.4: src/remax_minimal.py + remax_verl.py

### Task 4.5: lectures/08-vineppo.md (22 slides)
- MC advantage 估计

### Task 4.6: src/vineppo_minimal.py + vineppo_verl.py

### Task 4.7: lectures/09-reinforce-plus-plus.md (22 slides)
- OpenRLHF 简化版

### Task 4.8: src/reinforce_pp.py + reinforce_pp_verl.py

### Task 4.9: src/tests/test_variants_consistency.py
- 4 变体 minimal vs verl loss 数值验证

### Task 4.10: notebooks/06-09 四个

### Commit + Tag: `r1-variants`

---

## Phase 5: L10-L11 R1 复现潮（TinyZero + Open-R1）

### Task 5.1: lectures/10-tinyzero.md (24 slides)
- UC Berkeley $30 复现完整路径
- Countdown 任务设计
- Qwen-3B 涌现 aha moment 现象

### Task 5.2: src/tinyzero_reproduce.py
- 跟随作者 repo，简化版本

### Task 5.3: lectures/11-open-r1.md (26 slides)
- HuggingFace 三步走（SFT-distill / GRPO / reward 多样化）
- 350k Mixture-of-Thoughts 数据集

### Task 5.4: src/open_r1_demo.py
- 加载 350k MoT 子集（1k）+ 三步走简化

### Task 5.5: notebooks/10 + 11

### Commit + Tag: `r1-reproductions`

---

## Phase 6: L12 Spurious Rewards 警示

### Task 6.1: lectures/12-spurious-rewards.md (28 slides) ⚠️ **必看**
- Qwen 用随机/spurious reward 也能涨 21pt 的现象
- 模型依赖性问题（Qwen 之外可能失效）
- 隐含的教训：reward 涨不等于真涨

### Task 6.2: src/spurious_rewards_demo.py
- 复现关键现象：Qwen-base 用随机 binary reward 训 GRPO，看到 accuracy 涨
- 但用 LLaMA-base 同实验，accuracy 不涨（佐证模型依赖性）

### Task 6.3: notebooks/12-spurious-rewards.ipynb

### Commit + Tag: `r1-spurious-warning`

---

## Phase 7: L13 Capstone-A（教学轨）⭐⭐⭐ 必跑

### Task 7.1: lectures/13-capstone-r1-zero-track-A.md (28 slides)
- 完整 walkthrough：算法对照 + 训练 + 观察

### Task 7.2: src/r1_zero_track_a/
- 完整代码组织：
  - `data/countdown_3_data.py` — Countdown-3 数据生成 (5k train / 500 val)
  - `train_reinforce.py` — 算法 1: REINFORCE + mean baseline
  - `train_rloo.py` — 算法 2: RLOO k=8
  - `train_grpo.py` — 算法 3: GRPO k=8（关键）
  - `train_grpo_clip_higher.py` — 算法 4: GRPO + DAPO Clip-Higher
  - `eval_pipeline.py` — format / accuracy / length 评估
- 基座: GPT-2-medium (355M)
- 预期：format 5%→95%, accuracy 5%→15%, length 50→150

### Task 7.3: src/tests/test_track_a_pipeline.py
- 50 step 后 format reward 上升趋势

### Task 7.4: notebooks/13-capstone-r1-zero-track-A.ipynb
- 完整跑通展示 + 4 算法对照曲线

### Commit + Tag: `r1-capstone-track-a`

---

## Phase 8: L14 Capstone-B（挑战轨）选跑

### Task 8.1: lectures/14-capstone-r1-zero-track-B.md (24 slides)
- Qwen-1.5B + 4bit LoRA + GSM8K-tiny
- 期待看到 aha moment

### Task 8.2: src/r1_zero_track_b/
- `data/gsm8k_tiny.py` — GSM8K-tiny (500 train / 100 test)
- `train_grpo_dapo.py` — GRPO + DAPO
- `eval_aha_moment.py` — 检测 wait/recheck/let me reconsider 词频

### Task 8.3: 预训 ckpt 下载脚本（供学员跳过 4h 训练）

### Task 8.4: notebooks/14-capstone-r1-zero-track-B.ipynb
- 训完观察 aha moment

### Commit + Tag: `r1-capstone-track-b`（选作）

---

## Phase 9: L15 总结 + README

### Task 9.1: lectures/15-r1-takeaway.md (24 slides)
- 算法对照表（GRPO/RLOO/ReMax/VinePPO/REINFORCE++）
- 工程 trick 总结
- 与传统 RLHF 对比

### Task 9.2: notebooks/15-r1-takeaway.ipynb
- 综合可视化

### Task 9.3: README.md
- 15 方法横向表 + R1 工程清单 + 自测 14 题
- ⚠️ Spurious Rewards 警示突出显示

### Task 9.4: papers/ 补全

### Commit + Tag: `reasoning-r1` ⭐⭐⭐⭐⭐

---

## Phase 10: 收尾验证

### Task 10.1: 全部 notebook 执行（注意：capstone 跑完需 10h+）
### Task 10.2: 全部测试
### Task 10.3: Capstone-A 教学轨 format reward 95%+ + accuracy 提升
### Task 10.4: Capstone-B 挑战轨 aha 词频 ≥ 5%（选验）
### Task 10.5: verify_env.py PASS

### Final commit: `docs: reasoning-r1 README + complete`

---

## 完成验收清单

- [ ] WSL2 环境 setup 完成
- [ ] 15 lecture markdown
- [ ] 15 notebook 全跑（capstone 可分批）
- [ ] 5 个算法 (GRPO/RLOO/ReMax/VinePPO/REINFORCE++) minimal vs verl/trl 一致
- [ ] Capstone-A: format reward 95%+ + accuracy 提升（必）
- [ ] Capstone-B: aha 词频 ≥ 5%（选）
- [ ] verify_env.py 全 PASS
- [ ] tag `reasoning-r1` ⭐⭐⭐⭐⭐ 系列高峰

**预计 git commits**: ~35
**预计实施时长**: 18 hours（含 capstone 训练 10h+）
