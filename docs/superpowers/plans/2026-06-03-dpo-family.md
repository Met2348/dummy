# DPO Family Implementation Plan

**Goal**: 完整实现 DPO 家族 13 个偏好优化方法学习包（13 lecture + minimal/trl 两轨 + 7 测试 + 13 notebook + 6 方法对照 Capstone）

**Architecture**: 13 lectures = 11 主线 + 1 RainbowPO 统一 + 1 Capstone。两轨代码：minimal + trl（含 DPO/KTO/ORPO/CPO Trainer）。

**Tech Stack**: torch 2.13.0 cu130 / transformers 5.x / trl 0.13 / peft 0.19 / Qwen2.5-0.5B 基座

**Design 文档**: `docs/superpowers/specs/2026-06-03-dpo-family-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/dpo-family/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- trl + datasets + accelerate + peft + matplotlib + seaborn

### Task 1.3: environment/verify_env.py
- Part A: trl 各 Trainer init (DPO/KTO/ORPO/CPO)
- Part B: GPU
- Part C: DPOTrainer Qwen-0.5B 5-step smoke

### Task 1.4: src/common.py
- preference data loader (Anthropic-HH 1k)
- KTO 单边数据格式
- Qwen2.5-0.5B 加载 helpers

### Task 1.5: papers/ 12 个占位 + README

### Commit: `chore: dpo-family scaffold`

---

## Phase 2: L1 DPO 推导 + 强一致测试

### Task 2.1: lectures/01-dpo.md (28 slides)
- 5 步完整代数推导（从 PPO+KL+BT → DPO closed-form）
- 几何意义 + chosen/rejected log prob 关系

### Task 2.2: src/dpo_minimal.py
- 手写 DPO loss（一行核心代码）
- 训练 loop（reference model 冻结）

### Task 2.3: src/dpo_trl.py
- trl DPOTrainer 完整 demo
- Qwen-0.5B + Anthropic-HH 1k 子集

### Task 2.4: src/tests/test_dpo_loss_equivalence.py
- minimal vs trl loss < 1e-6（同输入）
- 100 step 训练后 chosen-rejected margin 增大

### Task 2.5: notebooks/01-dpo.ipynb
- 公式推导可视化 + chosen/rejected 概率漂移图

### Commit + Tag: `dpo-core`

---

## Phase 3: L2-L4 IPO + KTO + ORPO

### Task 3.1: lectures/02-ipo.md (22 slides)
- IPO 公式 + 与 DPO 的 RMSE vs cross-entropy 对比

### Task 3.2: src/ipo_minimal.py + ipo_trl.py
- `DPOTrainer(loss_type="ipo")` 调用

### Task 3.3: lectures/03-kto.md (24 slides)
- Kahneman-Tversky 损失公式
- 单边偏好数据格式

### Task 3.4: src/kto_minimal.py + kto_trl.py
- trl KTOTrainer

### Task 3.5: lectures/04-orpo.md (28 slides)
- Odds Ratio 推导 + **无 reference model** 关键证明

### Task 3.6: src/orpo_minimal.py + orpo_trl.py
- 验证 ORPO 训练不需要 ref model（显存对比）

### Task 3.7: src/tests/test_ipo_kto_orpo.py
- 3 个 loss 数值验证
- ORPO 无 ref model 显存对比

### Task 3.8: notebooks/02-04 三个

### Commit + Tag: `dpo-variants-1`

---

## Phase 4: L5-L7 SimPO + CPO + DPOP

### Task 4.1: lectures/05-simpo.md (28 slides)
- length-normalized loss 公式
- length norm 几何意义

### Task 4.2: src/simpo_minimal.py + simpo_trl.py
- trl `CPOTrainer(loss_type="simpo")`

### Task 4.3: lectures/06-cpo.md (22 slides)
- Contrastive + SFT 组合

### Task 4.4: src/cpo_minimal.py + cpo_trl.py

### Task 4.5: lectures/07-dpop.md (24 slides)
- DPO chosen 概率下降问题
- DPOP 修复机制

### Task 4.6: src/dpop_minimal.py
- 反例数据集（提供 known-good）
- 验证：DPO 训练后 chosen 概率下降，DPOP 上升

### Task 4.7: src/tests/test_simpo_cpo_dpop.py
- SimPO length norm 数值
- DPOP 反例验证

### Task 4.8: notebooks/05-07 三个

### Commit + Tag: `dpo-variants-2`

---

## Phase 5: L8-L11 Step-DPO + Iterative + Online + Nash

### Task 5.1: lectures/08-step-dpo.md (22 slides)
- 步骤级 DPO + GSM8K 玩具

### Task 5.2: src/step_dpo_minimal.py
- 手写 step-level DPO

### Task 5.3: lectures/09-iterative-dpo.md (20 slides)
- 多轮迭代 + OAIF 用 AI feedback

### Task 5.4: src/iterative_dpo.py
- 3 轮迭代框架

### Task 5.5: lectures/10-online-dpo.md (20 slides)
- on-policy 采样 + sDPO sequential

### Task 5.6: src/online_dpo.py + sdpo.py

### Task 5.7: lectures/11-nash-lhf.md (22 slides)
- Nash-MD / INPO / ONPO 系列

### Task 5.8: src/nash_lhf.py
- Nash-MD 简化版（mirror descent）

### Task 5.9: notebooks/08-11 四个

### Commit + Tag: `dpo-step-online`

---

## Phase 6: L12 RainbowPO 统一

### Task 6.1: lectures/12-rainbowpo.md (28 slides)
- 7 变体的 4 维超参（β / α / γ / length_norm）
- 统一公式推导

### Task 6.2: src/rainbowpo.py
- 统一 config wrapper
- 7 变体切换（通过超参组合）
- 一致性测试：RainbowPO(配置 X) ≡ trl Trainer(loss_type=X)

### Task 6.3: src/tests/test_rainbowpo_unification.py
- 7 配置下与 trl 对应 Trainer loss 一致

### Task 6.4: notebooks/12-rainbowpo.ipynb
- 7 变体在同基座同数据下的横向对比图

### Commit + Tag: `dpo-rainbow`

---

## Phase 7: L13 Capstone + README

### Task 7.1: lectures/13-capstone-dpo-comparison.md (28 slides)
- 6 方法对照 benchmark walkthrough
- 雷达图 + 工程选型

### Task 7.2: src/capstone_six_methods.py
- 同基座（Qwen2.5-0.5B）+ 同数据（Anthropic-HH 1k）
- 跑 DPO / IPO / KTO / ORPO / SimPO / CPO
- 输出：6 条 reward margin 曲线 + 雷达图

### Task 7.3: notebooks/13-capstone-dpo-comparison.ipynb
- 完整对照实验展示

### Task 7.4: README.md
- 13 方法横向表 + 数学推导 cheat sheet + 决策树 + 自测 14 题

### Task 7.5: papers/ 补全

### Commit + Tag: `dpo-family`

---

## Phase 8: 收尾验证

### Task 8.1: 全部 notebook 执行
### Task 8.2: 全部测试
### Task 8.3: Capstone 6 方法对照图完成
### Task 8.4: verify_env.py PASS

### Final commit: `docs: dpo-family README + complete`

---

## 完成验收清单

- [ ] 13 lecture markdown
- [ ] 13 notebook 全跑
- [ ] 6 个 DPO 变体 minimal vs trl loss < 1e-6
- [ ] Capstone 6 方法雷达图 + reward margin 曲线
- [ ] RainbowPO 统一性测试 PASS
- [ ] tag `dpo-family`

**预计 git commits**: ~28
**预计实施时长**: 14 hours
