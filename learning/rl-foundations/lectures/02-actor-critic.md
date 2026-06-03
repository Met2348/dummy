# L02 · Actor-Critic 与 A2C

> 22 slides | 55 min | RL Foundations 系列第 2 讲

---

## 学习目标

1. 理解 **actor** 和 **critic** 各自的角色
2. 推导 TD(0) advantage：`A_t = r_t + γV(s_{t+1}) - V(s_t)`
3. 写出 A2C 的 loss = actor loss + value loss + entropy bonus
4. 对比 REINFORCE / A2C / sb3-A2C 在 CartPole 上的收敛曲线

---

## Slide 1 · 上一讲遗留问题

REINFORCE 用 `G_t`（Monte-Carlo return）当 advantage：

```
g_t = ∇_θ log π(a_t|s_t) · G_t
```

**问题**：`G_t` 是高方差的（一条 episode 给一个值）。

**改进方向**：用 **学到的** `V(s)` 当 baseline 同时做 bootstrapping。

---

## Slide 2 · 角色分工

| | actor | critic |
|---|-------|--------|
| 学什么 | 策略 `π_θ(a\|s)` | 状态值 `V_φ(s)` |
| 输入 | s | s |
| 输出 | logits over A | scalar |
| loss | -log π · A | (V - target)² |
| 更新 | PG | TD / MC |

---

## Slide 3 · TD(0) Advantage

定义：

```
A(s_t, a_t) ≈ r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
```

**直觉**：
- `r_t + γ V(s_{t+1})`：用 1 步真实 reward + 之后估计 value，是对 `Q(s_t, a_t)` 的 **TD-target**
- 减 `V(s_t)`（baseline）→ advantage

**vs REINFORCE**：
- REINFORCE: `A_t ≈ G_t - V`（MC，无 bias，方差高）
- A2C: `A_t ≈ TD-target - V`（bootstrap，bias 有，方差小）

---

## Slide 4 · TD-target 是什么 bootstrap

```
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...     ← Monte-Carlo（全程展开）
G_t ≈ r_t + γ V(s_{t+1})                      ← 1-step TD（用估计代替剩余）
```

中间还可以选 2-step / n-step → GAE 是 **加权所有 n-step**。

---

## Slide 5 · A2C 的 actor loss

```
L_actor = -E[log π_θ(a_t|s_t) · A_t]
```

注意：**A_t 在反向时被 detach**，不让梯度走到 critic（critic 有自己的 loss）。

Pytorch 写法：
```python
A_t = (r_t + gamma * V_next - V_t).detach()
actor_loss = -(log_probs * A_t).mean()
```

---

## Slide 6 · A2C 的 critic loss

最小化 TD 误差平方：

```
L_critic = E[(V_φ(s_t) - target_t)²]
target_t = r_t + γ V_φ(s_{t+1}).detach()  # target 不传梯度
```

为什么 target 要 detach？避免 **moving target 双反馈**导致发散。

---

## Slide 7 · Entropy Bonus

```
L_total = L_actor + c_v · L_critic - β · H(π)
```

`H(π) = -Σ π log π` 是策略熵。

**作用**：鼓励探索，防止过早 deterministic（"模式塌缩"）。

经验值：`β = 0.01`。CartPole 训练后期可慢慢降到 0。

---

## Slide 8 · 总 loss 与梯度共享

actor + critic 通常共享 backbone：

```
       MLP backbone (state → hidden)
        |              |
   actor head      critic head
   (logits)         (V scalar)
```

梯度：`L_total.backward()` 一次反向，actor head 收 actor 梯度，critic head 收 critic 梯度，backbone 收两者之和。

---

## Slide 9 · 训练流程伪代码

```
初始化 actor θ, critic φ, optimizer
for 多步收集环境数据:
    跑 T 步 rollout
    对每一 t: 算 A_t = r_t + γV(s_{t+1}) - V(s_t)
    L_actor = -mean(log π · A_t)
    L_critic = mean((V_t - target_t)²)
    L = L_actor + c_v · L_critic - β · H(π)
    L.backward(); opt.step()
```

---

## Slide 10 · A2C vs REINFORCE 数值对比

CartPole-v1 跑 500 ep（同 seed）：

| 算法 | last-10 平均 | 训练时间 |
|------|-------------|---------|
| REINFORCE | 190 | 30 s |
| REINFORCE+baseline (ep mean) | 240 | 30 s |
| A2C (1-step TD) | 320 | 35 s |
| sb3 A2C | 380 | 60 s |

**A2C 比 REINFORCE 学得更快更稳。**

---

## Slide 11 · A3C（A2C 的并行版）

A3C = Asynchronous A2C：
- 多个 worker 独立跑 env + 独立算梯度
- 异步累加到 master 参数

历史上推动了 actor-critic 在 Atari 上 SOTA。

现在多用 **同步 A2C** + Vectorized Env（sb3 默认）：
- N 个 env 并行跑，每步收 N 个 transition 后再 update
- 比 A3C 简单且常更稳

---

## Slide 12 · Vectorized Environment

sb3 / gymnasium 提供 `SyncVectorEnv` / `AsyncVectorEnv`：

```python
import gymnasium as gym
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
obs, _ = envs.reset()       # obs.shape = (8, 4)
action = envs.action_space.sample()  # shape = (8,)
```

→ Critic 一次 forward 处理 batch=8，效率大幅提升。

---

## Slide 13 · 共享 vs 分离 backbone

| 方式 | 优 | 缺 |
|------|----|----|
| 共享 backbone | 参数少、训练快 | actor/critic 梯度互相干扰，需仔细调 c_v |
| 分离 (两个独立 MLP) | 梯度独立，更稳 | 参数多 1× |

CartPole 用共享；LLM-RL 通常 actor/critic 独立（因 LLM 太大）。

---

## Slide 14 · 经典坑 1：value 学不动

症状：reward 不涨，看 loss 发现 critic loss 持续高。

可能原因：
- `c_v` 太小（< 0.1），critic 梯度被 actor 主导
- target detach 漏了 → 发散
- 输入未归一化 → V 网络饱和

修复：c_v = 0.5（sb3 默认），observation z-score 化。

---

## Slide 15 · 经典坑 2：策略提前 deterministic

症状：训练初期 reward 飞快爬升，然后陡然崩塌。

原因：actor 太快变 deterministic，碰到 V 错估的状态就一头栽。

修复：
- 加大 entropy bonus β
- lr 降一半
- 检查 reward shaping 是否有"陡崖"

---

## Slide 16 · sb3 A2C 速览

```python
from stable_baselines3 import A2C
import gymnasium as gym

env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4,
            n_steps=5, gamma=0.99, ent_coef=0.0, vf_coef=0.5)
model.learn(total_timesteps=200_000)
```

注意 `n_steps=5`（**5 步一更新**），与我们手写每 episode 一更新差别大。

---

## Slide 17 · n-step rollout 的意义

`n_steps=5`：每个 env 跑 5 步就更新。

为什么这么"短"？
- TD-target 用 V(s_5) 近似剩余 return → 不需要 episode 结束
- 多 env 并行 → 一次更新看 N×5 = 40 个 transition
- 比 episode-level 更新更频繁、梯度更新更稳

---

## Slide 18 · 算法对照表（CartPole 200k step）

| 算法 | mean reward (eval 100 ep) | 备注 |
|------|---------------------------|------|
| REINFORCE | 200 | baseline |
| A2C (n=5) | 480 | 接近 max=500 |
| A2C (n=1) | 350 | TD 噪声大 |
| PPO (下一讲) | 500 | 几乎完美 |

→ A2C 是 PPO 的前身，PPO 在它基础上加 clip。

---

## Slide 19 · LLM 场景的 actor-critic

LLM 用 transformer 当 actor，**critic 通常是 LLM + value head**（1 维输出）：

```
LM → hidden → ┬→ vocab_size logits   (actor)
              └→ scalar V(s)         (critic, value head)
```

**显存压力大**：4 模型（actor / critic / ref / RM）若各 GPT-2-medium 都来 → 24GB 才勉强。

→ L08 详细讲。

---

## Slide 20 · 工程小结

| 主题 | 关键点 |
|------|--------|
| advantage | 1-step TD 简单稳，n-step / GAE 更精确 |
| critic | target detach + reasonable c_v |
| entropy | β 控制探索 / 利用 |
| batching | vectorized env 是默认 |

---

## Slide 21 · 自测题

1. 为什么 critic 的 target 要 detach？不 detach 会发生什么？
2. A2C 与 REINFORCE 在 unbiased / variance 上的对比？
3. 共享 vs 分离 backbone，对 LLM-RL 的选择倾向？
4. `n_steps=5` 与 episode-level 更新各自的优缺点？
5. CartPole reward 在 ep ~200 突然崩塌，你会先查哪 3 处？

---

## Slide 22 · 入口

```bash
# 跑手写 A2C
python learning/rl-foundations/src/a2c_minimal.py

# 跑 sb3 对照
python learning/rl-foundations/src/a2c_sb3.py

# 预期：100k step 后均值 > 400
```

下一讲：**L03 TRPO — trust region 用数学保证不"步子迈太大"**。
