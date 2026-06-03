# RL Foundations 学习专题 — 设计文档

> **承接**: prompt-tuning-family + lora-family + adapter-tuning-family (PEFT 三大主线完结)
> **本专题**: RL for LLM 系列第一站 — RL 算法基础
> **战略地位**: 为后续 6 个 RL/对齐/推理专题打底
> **总体规划**: `C:\Users\ericp\.claude\plans\partitioned-squishing-stream.md`

---

## 1. 专题定位

RL Foundations 是 RL for LLM 系列的"地基"。覆盖经典 RL 算法演化（REINFORCE → A2C → TRPO → PPO → GAE），结尾用 PPO 跑 GPT-2 玩具 RL 实验，为专题 2 RLHF 三段管线做准备。

### 1.1 为什么 PEFT 出身者需要 RL 速成
1. **范式不同**：PEFT 改 model（θ_LM），RL 改 distribution shape (p(y|x))
2. **训练循环不同**：PEFT 是监督学习，RL 是 rollout + reward + update
3. **失败模式不同**：PEFT 看 loss，RL 看 reward 曲线 + KL 漂移 + 生成质量
4. **工程复杂度不同**：PEFT 1 model 训，RL 通常 4 model 并存（actor/critic/ref/RM）

### 1.2 本专题的"双面性"
- **理论 50%**：MDP 框架、policy gradient 推导、PPO clip 几何意义
- **实战 50%**：CartPole 玩具 + GPT-2 + IMDb 情感奖励 RL

---

## 2. 方法清单（12 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **REINFORCE** | 1992 | Williams | Policy gradient 起源，∇J = E[∇log π·R] |
| 2 | **REINFORCE w/ baseline** | 1992 | Williams | 减方差 baseline (V(s)) |
| 3 | **Actor-Critic (AC)** | 2000 | Sutton | 联合 actor + critic |
| 4 | **A2C** | 2016 | Mnih (Atari) | 同步 advantage actor-critic |
| 5 | **A3C** | 2016 | Mnih | 异步多 worker A2C |
| 6 | **TRPO** | 2015 | Schulman | trust region + KL 约束 |
| 7 | **PPO-Clip** | 2017 | Schulman | clip ratio 替代 trust region（**核心**）|
| 8 | **PPO-KL Penalty** | 2017 | Schulman | adaptive KL penalty 变体 |
| 9 | **GAE** | 2015 | Schulman | λ-return 优势估计 |
| 10 | **Entropy Bonus** | 2016+ | — | 鼓励探索的辅助 loss |
| 11 | **PPO for LLM** | 2020+ | — | token-level reward + KL ref penalty |
| 12 | **KL Adaptive Control** | 2020+ | OpenAI | RLHF 中 KL 系数动态调整 |

---

## 3. Lecture 结构（12 篇 = 10 主线 + 2 capstone）

| Lecture | 主题 | 主方法 | 时长 |
|---------|------|--------|------|
| **L1** MDP + Policy Gradient | REINFORCE | 60 min |
| **L2** Actor-Critic | A2C / A3C | 60 min |
| **L3** TRPO | TRPO + 自然梯度 | 90 min（含数学推导）|
| **L4** PPO 核心 | PPO-Clip / KL-Penalty | 90 min |
| **L5** GAE 优势估计 | GAE | 60 min |
| **L6** PPO 工程 trick | clip range, entropy, value clip, orth init | 60 min |
| **L7** CartPole 实战 | gymnasium + sb3 + 手写 PPO | 90 min |
| **L8** PPO → LLM 首次接触 | token-level reward, KL penalty, 4-model | 90 min |
| **L9** GPT-2 玩具 RL | IMDb 情感 reward 实战 | 90 min |
| **L10** RL 陷阱合集 | reward hacking, KL 崩, lr 敏感 | 60 min |
| **L11** Capstone：GPT-2 + IMDb PPO | 完整 pipeline + 观察 | 90 min |
| **L12** 总结 + 引出 RLHF | 为什么需要 RM | 30 min |

**总学时**: 12 lecture × 平均 70 min + 5h notebook ≈ 14 hours

---

## 4. Lecture 模板（PPT-style，每篇 18-26 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（这个方法解决什么问题）
## Slide 3-4: 核心公式（policy gradient theorem / PPO clip 等）
## Slide 5-7: 直觉解释 / 几何意义（图解）
## Slide 8-12: 数学推导（trust region / GAE 等）
## Slide 13-16: 代码逐行（minimal vs sb3/trl）
## Slide 17-19: 实验观察（CartPole reward 曲线 / GPT-2 KL 漂移）
## Slide 20-22: 陷阱与警示（这个方法什么时候不 work）
## Slide 23-25: 思考题 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | minimal | 库 1 (sb3) | 库 2 (trl) | 备注 |
|------|---------|-----------|-----------|------|
| REINFORCE | ✅ 手写 | — | — | CartPole 玩具 |
| A2C | ✅ 手写 | ✅ sb3.A2C | — | CartPole |
| PPO (CartPole) | ✅ 手写 | ✅ sb3.PPO | — | 强一致对照 |
| GAE | ✅ 手写 | (sb3 内置) | — | 独立模块 |
| PPO for LLM | ✅ 手写 | — | ✅ trl.PPOTrainer | GPT-2 实战 |

**目录约定**:
- `{method}_minimal.py` — 手写最小实现
- `{method}_sb3.py` — stable-baselines3 对照（仅 CartPole）
- `ppo_gpt2_trl.py` — trl 对照（LLM 实战）

---

## 6. 一致性测试

```python
# RL 系列的"一致性"≠ PEFT 的 bit-exact
def test_loss_formula():       # minimal vs sb3 loss < 1e-6（同输入）
def test_gradient_cosine():    # 同 seed 第一步梯度 cosine > 0.999
def test_reward_trend():       # 200 步 CartPole reward 上升趋势
def test_kl_stability():       # GPT-2 PPO KL 不爆 (<10)
```

**强一致**: REINFORCE / A2C / PPO-CartPole vs sb3
**弱一致**: PPO-LLM vs trl（实现细节差，趋势一致即可）

---

## 7. Notebook 结构（12 个）

每个 lecture 一个 ipynb：
1. import + 环境/模型加载
2. 公式可视化（policy gradient / clip 函数图）
3. minimal 实现 step-by-step
4. mini training（200 step CartPole / 100 step GPT-2）
5. 库对照（sb3 / trl）
6. 关键指标可视化（reward 曲线 / KL 漂移 / advantage 分布）
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (Windows native, 复用 PEFT cu130 环境)
torch>=2.5+cu130
transformers>=5.0
trl>=0.13
peft>=0.13                # 用于 LoRA-adapted PPO（节省显存）
gymnasium>=0.29
stable-baselines3>=2.3
matplotlib, seaborn
tensorboard
datasets                  # IMDb
```

**verify_env.py 三段式**:
- Part A: 基础（torch + transformers + trl）+ gymnasium import
- Part B: GPU + sm_120 (RTX 5090)
- Part C: trl PPOTrainer smoke test (GPT-2 + 5 step)

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `rl-base-pg` | L1-L2: REINFORCE + A2C | 4 |
| `rl-base-ppo` | L3-L6: TRPO + PPO + GAE + tricks | 6 |
| `rl-base-cartpole` | L7: CartPole 实战完成 | 2 |
| `rl-base-llm` | L8-L10: PPO for LLM + 陷阱 | 4 |
| `rl-foundations` | L11-L12: Capstone + 总结 | 3 |

---

## 10. 跨专题衔接

### 上游
- 无（系列首站）
- PEFT 知识假设：用户会 GPT-2 + transformers + tokenizer

### 下游
- 专题 2 RLHF：复用 PPO for LLM 框架，加 BT-RM
- 专题 3 DPO：与 PPO+BT 对比，看 DPO 怎么"短路"
- 专题 5 R1：GRPO 是 PPO 的 group baseline 变体

### 跨专题对照表预留位（README）
| RL 算法 | 适用场景 | 应用专题 |
|---------|---------|---------|
| PPO | 通用 RLHF | 专题 2 |
| RLOO / GRPO | LLM 推理 RL | 专题 5 |
| ReMax | 节省 critic | 专题 5 |
| Nash-MD | Nash learning | 专题 3 |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| PPO 超参极敏感 | 高 | 中 | 提供 known-good config + sensitivity 实验图 |
| 4-model 24GB OOM | 中 | 高 | GPT-2-small (124M) + shared ref/RM + bf16 |
| CartPole 训练不收敛 | 低 | 低 | seed 固定 + 200 step 足够 |
| GPT-2 PPO KL 爆炸 | 中 | 高 | KL adaptive + clip 实战教程 |
| trl 与 transformers 版本冲突 | 中 | 中 | 独立 venv（参考 adapter 系列做法）|

---

## 12. 论文 / 资料占位

```
papers/
├── 01-williams-1992-reinforce.md
├── 02-mnih-2016-a3c.md
├── 03-schulman-2015-trpo.md
├── 04-schulman-2017-ppo.md
├── 05-schulman-2015-gae.md
├── 06-engstrom-2020-ppo-implementation-matters.md
├── 07-zheng-2023-ppo-llm-secrets.md   # "The N Implementation Details of RLHF with PPO"
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-03-rl-foundations.md` 的 7 个 Phase 推进（L1-L2 / L3-L4 / L5-L6 / L7 / L8-L9 / L10-L11 / L12+README），每 Phase 1 commit + 部分 tag。
