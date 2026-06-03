# RL Foundations Implementation Plan

**Goal**: 完整实现 RL 基础学习包（12 lecture + 12 minimal/sb3/trl + 6 测试 + 12 notebook）

**Architecture**: 12 lectures = 10 主线 + 2 capstone。三轨代码：minimal + stable-baselines3（仅 CartPole）+ trl（仅 LLM）。复用 PEFT cu130 nightly Windows native 环境。

**Tech Stack**: torch 2.13.0.dev cu130 / transformers 5.x / trl 0.13 / gymnasium 0.29 / stable-baselines3 2.3 / peft 0.19

**Design 文档**: `docs/superpowers/specs/2026-06-03-rl-foundations-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/rl-foundations/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + transformers + trl + gymnasium + sb3 + peft + matplotlib + tensorboard

### Task 1.3: environment/verify_env.py
- Part A: torch + transformers + trl + gymnasium import
- Part B: GPU + sm_120
- Part C: trl PPOTrainer smoke (GPT-2 + 5 step)

### Task 1.4: src/common.py
- 复用 PEFT 系列的 freeze / get_in_out_dims
- 新增：compute_advantage / log_prob / kl_divergence helpers

### Task 1.5: papers/ 7 个占位 + README index

### Commit: `chore: rl-foundations scaffold`

---

## Phase 2: L1-L2 Policy Gradient + Actor-Critic

### Task 2.1: lectures/01-mdp-policy.md (24 slides)
- MDP 框架、policy gradient theorem 推导、REINFORCE 算法、variance 问题

### Task 2.2: src/reinforce_minimal.py
- `REINFORCEAgent(state_dim, action_dim)`
- CartPole 训练 loop（200 episode）
- 显示 reward 曲线

### Task 2.3: lectures/02-actor-critic.md (22 slides)
- A2C 推导、advantage、actor-critic 联合训练

### Task 2.4: src/a2c_minimal.py
- `A2CAgent` 手写
- `a2c_sb3.py` 用 stable-baselines3 对照

### Task 2.5: src/tests/test_reinforce_cartpole.py + test_a2c_cartpole.py
- 200 episode reward 上升趋势

### Task 2.6: notebooks/01-mdp-policy.ipynb + 02-actor-critic.ipynb
- 公式可视化 + CartPole 曲线对比

### Commit + Tag: `rl-base-pg`

---

## Phase 3: L3-L6 TRPO + PPO + GAE + tricks

### Task 3.1: lectures/03-trpo.md (28 slides)
- trust region 推导、自然梯度、Fisher 信息矩阵、TRPO 算法步骤

### Task 3.2: src/trpo_minimal.py
- 简化版（不优化共轭梯度，直接 backward + line search）
- CartPole 训练

### Task 3.3: lectures/04-ppo-core.md (28 slides)
- PPO-Clip 推导、surrogate loss、clip 几何意义
- KL-Penalty 变体

### Task 3.4: src/ppo_minimal.py + ppo_sb3.py
- 手写 PPO（clip + value loss + entropy bonus）
- sb3 对照

### Task 3.5: lectures/05-gae.md (22 slides)
- λ-return 推导、bias-variance trade-off、GAE 算法

### Task 3.6: src/gae.py
- 独立模块（被 ppo_minimal 调用）
- 数值测试

### Task 3.7: lectures/06-ppo-tricks.md (20 slides)
- clip range / entropy / value clip / orthogonal init / advantage norm

### Task 3.8: src/ppo_tricks_ablation.py
- 5 trick 单独消融

### Task 3.9: src/tests/test_ppo_consistency.py
- minimal vs sb3 loss < 1e-6（同输入）
- 200 step CartPole reward 趋势

### Task 3.10: notebooks/03-trpo.ipynb + 04-ppo-core.ipynb + 05-gae.ipynb + 06-ppo-tricks.ipynb

### Commit + Tag: `rl-base-ppo`

---

## Phase 4: L7 CartPole 实战

### Task 4.1: lectures/07-cartpole-lab.md (20 slides)
- gymnasium 环境、render、reward shaping、训练观察

### Task 4.2: src/cartpole_full.py
- 手写 PPO + sb3 双轨完整 CartPole 训练
- TensorBoard 日志

### Task 4.3: notebooks/07-cartpole-lab.ipynb
- 完整交互式实验

### Commit + Tag: `rl-base-cartpole`

---

## Phase 5: L8-L10 PPO for LLM + 陷阱

### Task 5.1: lectures/08-ppo-for-llm.md (26 slides)
- token-level reward / KL ref penalty / 4-model 协同 / value head 加在 LM 上

### Task 5.2: src/ppo_gpt2_minimal.py
- 手写 PPO for GPT-2（核心）
- 包含 4 model: actor / critic / ref / RM
- KL adaptive control

### Task 5.3: lectures/09-toy-rl-llm.md (24 slides)
- 句长 reward / IMDb sentiment reward / KL 控制实战

### Task 5.4: src/ppo_gpt2_trl.py
- trl PPOTrainer 对照

### Task 5.5: src/sentiment_reward.py
- BERT-sentiment 当 reward model

### Task 5.6: lectures/10-rl-pitfalls.md (22 slides)
- reward hacking / KL 崩 / lr 敏感性 / mode collapse

### Task 5.7: src/tests/test_gpt2_ppo.py
- 100 step 后 reward 上升 + KL < 10

### Task 5.8: notebooks/08-ppo-for-llm.ipynb + 09-toy-rl-llm.ipynb + 10-rl-pitfalls.ipynb

### Commit + Tag: `rl-base-llm`

---

## Phase 6: L11-L12 Capstone + 总结

### Task 6.1: lectures/11-capstone-ppo-llm.md (24 slides)
- 完整 pipeline 走一遍：数据 → 模型 → 训练 → 评估

### Task 6.2: src/capstone_imdb_ppo.py
- GPT-2-medium + IMDb 情感 reward
- 256 step batch 32
- 完整 TensorBoard

### Task 6.3: lectures/12-summary-to-rlhf.md (16 slides)
- 总结 RL 基础、引出"为什么 RL+LLM 需要 RM"
- 衔接专题 2

### Task 6.4: notebooks/11-capstone-ppo-llm.ipynb + 12-summary.ipynb

### Task 6.5: README.md（标准模板）
- 12 方法横向表
- 学习路径
- 横向对比（algorithm / variance / wall time）
- 自测 10 题
- 跨专题衔接

### Task 6.6: papers/ 补全 + 总 index

### Commit + Tag: `rl-foundations`

---

## Phase 7: 收尾验证

### Task 7.1: 全部 notebook 执行 (`jupyter nbconvert --execute --inplace`)

### Task 7.2: 全部测试 (`pytest learning/rl-foundations/src/tests/`)

### Task 7.3: Capstone IMDb reward 提升 ≥ 30% 验证

### Task 7.4: verify_env.py 跑通

### Final commit: `docs: rl-foundations README + complete`

---

## 完成验收清单

- [ ] 12 lecture markdown 完整
- [ ] 12 notebook 全部执行成功
- [ ] minimal vs sb3 一致性 PASS
- [ ] Capstone IMDb reward 提升 ≥ 30%
- [ ] verify_env.py Part A/B/C 全 PASS
- [ ] tag `rl-foundations`

**预计 git commits**: ~25
**预计实施时长**: 14 hours（学习 + 实施）
