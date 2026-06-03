# L01 · MDP 与 Policy Gradient 起源

> 24 slides | 60 min | RL Foundations 系列第 1 讲

---

## 学习目标

1. 写出 MDP 五元组 `(S, A, P, R, γ)`，理解每个符号
2. 推导 **policy gradient theorem**（PG 定理）
3. 写出 **REINFORCE** 算法（伪代码 + Pytorch 10 行版）
4. 解释 PG 的 **高方差** 来源 + 三种降方差技巧

---

## Slide 1 · 为什么从 RL 进入 LLM 对齐

PEFT 系列改的是 **模型本身**（参数）：

```
θ_LM → θ_LM + Δ           （Adapter / LoRA / Prompt）
```

RL 系列改的是 **采样分布与轨迹**：

```
p(y|x; θ_LM) → p(y|x; θ_LM, R, V)
```

`R` 是 reward，`V` 是 value head。后续两个专题（RLHF + DPO）都是这个改法的不同实现。

---

## Slide 2 · 一句话定义 MDP

> **Markov Decision Process** = 决策者在一系列状态间转移、每次转移获得 reward、目标是最大化未来 reward 总和。

数学描述：五元组

| 符号 | 含义 |
|------|------|
| `S` | 状态空间 |
| `A` | 动作空间 |
| `P(s'\|s, a)` | 转移概率（环境给定） |
| `R(s, a)` | 即时奖励 |
| `γ ∈ [0, 1]` | 折扣因子（trade-off 即时 vs 长远） |

LLM 类比：
- 状态 `s` = 已生成的 token 前缀
- 动作 `a` = 下一个 token
- 转移 `P` = 确定性（拼接）
- 奖励 `R` = 通常仅在终末由 RM 给

---

## Slide 3 · Policy 与 Trajectory

策略 `π(a|s)`：从状态到动作的（条件）概率分布。

一条 trajectory（轨迹）：

```
τ = (s₀, a₀, r₀, s₁, a₁, r₁, ..., s_T, a_T, r_T)
```

轨迹概率：

```
p(τ; π) = ρ(s₀) ∏_{t=0..T} π(a_t|s_t) P(s_{t+1}|s_t, a_t)
```

未来折扣 return：

```
G(τ) = Σ_{t=0..T} γ^t r_t
```

---

## Slide 4 · 目标函数

最大化期望 return：

```
J(π) = E_{τ ~ p(τ;π)} [G(τ)]
```

**关键**：`J` 是 `π` 的函数 → 我们要对 **策略参数 θ** 求梯度。

---

## Slide 5 · 策略参数化

`π_θ(a|s)` 是 NN 输出的概率（如 softmax）。

例：CartPole（2 个动作）
```python
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
    def forward(self, s): return self.fc(s)   # logits
```

`π_θ(a|s) = softmax(logits)[a]`

---

## Slide 6 · Policy Gradient Theorem（推导起点）

要算 `∇_θ J(π_θ)`，朴素思路碰壁：

```
∇_θ E[G] = ∇_θ ∫ p(τ;θ) G(τ) dτ
        = ∫ [∇_θ p(τ;θ)] G(τ) dτ   ← 这里 G(τ) 与 θ 无关（环境给的）
```

但是 `∇_θ p(τ;θ)` 不是个 distribution，无法 Monte-Carlo 估计。

---

## Slide 7 · Log-Derivative Trick

```
∇_θ p(τ;θ) = p(τ;θ) · ∇_θ log p(τ;θ)
```

代入：

```
∇_θ E[G] = ∫ p(τ;θ) [∇_θ log p(τ;θ)] G(τ) dτ
        = E_{τ ~ π_θ} [G(τ) · ∇_θ log p(τ;θ)]
```

**这一行是 RL 的"皇冠公式"**。Monte-Carlo 可估。

---

## Slide 8 · 展开 log p(τ)

```
log p(τ;θ) = log ρ(s₀) + Σ_t log π_θ(a_t|s_t) + Σ_t log P(s_{t+1}|s_t, a_t)
```

对 `θ` 求导，**初始分布 ρ 与转移 P 不依赖 θ**，直接消去：

```
∇_θ log p(τ;θ) = Σ_t ∇_θ log π_θ(a_t|s_t)
```

---

## Slide 9 · Policy Gradient Theorem（完整形式）

```
∇_θ J(π_θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · G(τ) ]
```

**MC 估计**：跑一条轨迹 `τ`，按

```
g = Σ_t ∇_θ log π_θ(a_t|s_t) · G(τ)
```

更新 `θ ← θ + α g`。这就是 REINFORCE。

---

## Slide 10 · REINFORCE 伪代码

```
初始化 θ
重复:
  按 π_θ 跑一条 episode，得 (s_0,a_0,r_0,...,s_T,a_T,r_T)
  计算 G_t = Σ_{k≥t} γ^(k-t) r_k   （从 t 时刻起的折扣 return）
  对每个 t:
    g_t = ∇_θ log π_θ(a_t|s_t) · G_t
  θ ← θ + α · Σ_t g_t
```

注意：用 `G_t`（从 t 起）而非 `G(τ)`（全程总和）—— 因为 t 之前的动作与未来 reward 无因果关系，这是 **causality** 简化。

---

## Slide 11 · 一个能跑的 Pytorch 实现（10 行）

```python
log_probs, rewards = [], []
state, _ = env.reset()
done = False
while not done:
    logits = policy(torch.as_tensor(state))
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_probs.append(dist.log_prob(action))
    state, r, terminated, truncated, _ = env.step(action.item())
    rewards.append(r); done = terminated or truncated

returns = compute_returns(rewards, dones=[False]*(len(rewards)-1)+[True], gamma=0.99)
loss = -sum(lp * G for lp, G in zip(log_probs, returns))
loss.backward(); opt.step()
```

---

## Slide 12 · 高方差 — REINFORCE 的"原罪"

观察：一条 episode 的 `G_t` 抖动剧烈 → `g_t` 抖动剧烈 → 学习曲线"忽上忽下"。

直觉：`G_t` 越大方差越大；而 `∇_θ log π` 的 magnitude 又跟它无关地缩放。

例：CartPole，一条 200 步 episode return = 200；另一条 5 步 return = 5。方差比 40×。

---

## Slide 13 · 降方差技巧 1：Causality

已用过：把 `G(τ)` 换成 `G_t`（仅 t 之后的 reward）。

**收益**：每个 `t` 的"乘子"只看到自身因果链产生的 reward 总和，方差小很多。
**代价**：无。

---

## Slide 14 · 降方差技巧 2：Baseline

观察：从 PG 公式减去 **任何不依赖 a_t 的项** `b(s_t)`，期望仍然不变：

```
E[∇_θ log π(a_t|s_t) · b(s_t)] = b(s_t) · ∇_θ E[1] = 0
```

但方差变小（如 `b` 与 `G_t` 相关）。

→ `g_t = ∇_θ log π · (G_t - b(s_t))`

最佳 baseline 是 **状态值函数** `V(s_t)`。

---

## Slide 15 · 降方差技巧 3：Advantage

定义：

```
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
```

意思：动作 `a_t` 相对于平均水平有多好。

PG 变成：

```
g = E [Σ_t ∇_θ log π(a_t|s_t) · A(s_t, a_t)]
```

如何估 `A`？三种思路：
1. MC：`A_t ≈ G_t - V(s_t)`（REINFORCE with baseline）
2. TD：`A_t ≈ r_t + γV(s_{t+1}) - V(s_t)`（actor-critic，下一讲）
3. GAE：1+2 加权平均（L05 主题）

---

## Slide 16 · CartPole 实战 1：环境

```python
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=42)
# obs.shape = (4,), 包括 cart pos / vel / pole angle / ang vel
# action ∈ {0, 1}（向左/向右推）
# reward = 每步 +1，倒下或越界终止
# max episode = 500 step
```

---

## Slide 17 · CartPole 实战 2：训练曲线

预期：
- 0-50 ep：reward 平均 ~20（随机）
- 50-150 ep：开始爬升
- 200 ep：均值 ≥ 195（被认为"solved"）

观察：
- 单 episode reward 抖动剧烈（高方差）→ 用 **moving avg 10 ep** 平滑

---

## Slide 18 · 与 LLM 的第一处对接

CartPole 玩具：

```
obs = 4-dim state
action = {0, 1}
reward = 每步 1
```

LLM 类比：

```
obs = 当前 token 序列 (变长)
action = 下一个 token (vocab_size 维)
reward = 通常在序列结束时给
```

→ 算法不变（PG 定理），但**变长**带来工程麻烦：每个 batch 长度不一，要 masking。

后续 L08 详细讲。

---

## Slide 19 · 与 Supervised Learning 的关键差异

| | SL | RL |
|---|----|----|
| label | 有金标 y | 无金标，仅 reward |
| 数据 | i.i.d. | 自相关（轨迹）|
| 梯度估计 | 解析 | Monte-Carlo（高方差） |
| 学习目标 | 减误差 | 增 return |

⚠️ RL **没有 train/val split 概念**（你的策略 改 数据分布）。

---

## Slide 20 · 折扣因子 γ 怎么选

| 任务 | γ |
|------|---|
| CartPole | 0.99（≈ 视野 ~100 步）|
| Atari | 0.99 |
| 围棋 | 1.0（情节最终结束）|
| LLM 生成 | 1.0（不折扣）或 0.95-0.99（强调早期 token）|

**经验**：γ 越接近 1，目标越难学但越接近真实长期 return。

---

## Slide 21 · 算法稳定性技巧速览

PG 训练经常崩，关键 trick：

| trick | 作用 |
|-------|------|
| advantage normalization | 减均值除标准差，降梯度噪声 |
| entropy bonus | 加 `+ β H(π)`，防止过早收敛到 deterministic |
| clipped reward | r ∈ [-1, 1]，防止极端值主导 |
| smaller lr (1e-4 vs 1e-3) | PG 对 lr 极敏感 |

---

## Slide 22 · 本讲与后续的衔接

**L02 Actor-Critic**：用 critic 学 `V(s)` 当 baseline → 进一步降方差。

**L03 TRPO / L04 PPO**：在 PG 框架上加 **trust region** / **clip**，让大 batch 多 epoch 更新不崩。

**L08 PPO for LLM**：把 GPT-2 当 actor，外加 critic + ref + RM，做 4 模型协同。

---

## Slide 23 · 自测题

1. 推导 PG 定理时，为什么 `∇_θ log P(s_{t+1}|s_t, a_t)` 消去？
2. REINFORCE 与 Actor-Critic 的本质差别？
3. 若把 `G_t` 改成 `G(τ)`（不用 causality），梯度估计是否仍 unbiased？方差呢？
4. 减 baseline `b(s)` 为何不改变期望？给出数学证明。
5. CartPole 训练曲线 200 ep 均值 < 50，你会先检查哪 3 处？

---

## Slide 24 · 实战入口

```bash
# 学完本讲跑一遍
python learning/rl-foundations/src/reinforce_minimal.py

# 预期：
# Ep 200: avg(last 10) = 180~250  → solved
# Loss 持续下降 + log_std 缓慢减小
```

下一讲：**L02 Actor-Critic — 给 PG 装一个 critic**。
