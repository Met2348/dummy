# L07 · CartPole 完整实战

> 20 slides | 50 min | RL Foundations 系列第 7 讲

---

## 学习目标

1. 熟悉 gymnasium API（reset/step/render/wrappers）
2. 用 TensorBoard 看完整训练过程
3. 把前 6 讲的算法（REINFORCE/A2C/TRPO/PPO/sb3-PPO）放在同一图上对比
4. 看到 PPO 单卡 200k step 把 CartPole-v1 推到接近满分

---

## Slide 1 · gymnasium API 基础

```python
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

注意：gym → gymnasium（2023 改名后）；`terminated` 与 `truncated` 分离。

---

## Slide 2 · CartPole-v1 规格

| 项 | 值 |
|----|---|
| obs | 4-dim (pos, vel, angle, ang_vel) |
| action | 2 (left, right) |
| reward | 每步 +1 |
| ep max | 500 步 |
| terminate | 倒下或越界 |
| 算"解决" | last 100 ep 均 ≥ 195（v0），v1 是 ≥ 475 |

---

## Slide 3 · Vectorized Env

```python
envs = gym.vector.SyncVectorEnv([
    lambda i=i: gym.make("CartPole-v1") for i in range(8)
])
obs, _ = envs.reset()  # shape (8, 4)
actions = envs.action_space.sample()  # shape (8,)
obs, rewards, term, trunc, _ = envs.step(actions)
```

→ 一次 step 跑 8 个 env，rollout 效率 8×。

---

## Slide 4 · Wrapper：observation / reward normalization

sb3 的 `VecNormalize`：
- 维护 running mean/std 的 obs
- 每个 step `(obs - mean) / std` 化
- 同时 normalize/clip reward

```python
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
```

→ CartPole 用不用 normalize 都行，Mujoco/Atari 几乎必须用。

---

## Slide 5 · TensorBoard 日志骨架

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/cartpole_ppo")
# 在 iter 循环里
writer.add_scalar("rollout/ep_mean", mean_R, global_step)
writer.add_scalar("loss/L_clip", L_clip.item(), global_step)
writer.add_scalar("loss/L_vf", L_vf.item(), global_step)
writer.add_scalar("policy/entropy", entropy.item(), global_step)
writer.add_scalar("policy/kl_approx", mean_kl, global_step)
```

启动：`tensorboard --logdir runs/`，浏览器 http://localhost:6006

---

## Slide 6 · 关键观察 1：ep_mean 上升曲线

- 0-20k step：随机阶段，mean ≈ 22
- 20-50k step：策略开始有效，mean 爬到 ~150
- 50-100k step：陡升到 400+
- 100k+：稳在 480-500

→ "陡升"是 RL 训练典型现象。

---

## Slide 7 · 关键观察 2：KL approx

- 健康范围：每 iter < 0.05
- KL > 0.1：触发 PPO clip 频繁，进步停滞
- KL 突跌到 0：模式塌缩信号（策略 deterministic 了）

→ 用 `kl_approx = (r-1-log r)` 监控（cheap）。

---

## Slide 8 · 关键观察 3：entropy

- 初始：≈ 0.69 (log 2)
- 末期：≈ 0.05~0.1（学会了，但还有探索）
- 跌到 0：deterministic，可能 collapse

→ 加 ent_coef=0.01 防止 entropy 过早塌缩。

---

## Slide 9 · 关键观察 4：L_clip 与 L_vf

- L_clip：通常负值，逐渐趋近 0（学不动了）
- L_vf：critic 平方误差，应稳定下降然后趋于平稳

如果 L_vf 不下降 → critic 学不动 → advantage 全错 → actor 也学不动。

---

## Slide 10 · 5 算法横向（同 100k step）

```
                 last 100 ep mean reward
REINFORCE         200
REINFORCE+bsl     240
A2C n=5           480
TRPO (简化)       490
PPO minimal       500 ⭐
sb3 PPO           500
```

把 5 条曲线画在同一张图上：横轴 env steps，纵轴 ep_mean_reward。

---

## Slide 11 · 复现实战代码

`src/cartpole_full.py` 把 5 个算法放在一个文件，可命令行切换：

```bash
python src/cartpole_full.py --algo reinforce --total-steps 100_000
python src/cartpole_full.py --algo a2c --total-steps 100_000
python src/cartpole_full.py --algo ppo --total-steps 100_000
python src/cartpole_full.py --algo sb3_ppo --total-steps 100_000
```

每个跑完写 TensorBoard，最后绘 5 条曲线。

---

## Slide 12 · 训练时间参考（5090 CPU+GPU）

| 算法 | 100k step wall time |
|------|-------------------|
| REINFORCE | 40 s |
| A2C n=5 | 60 s |
| TRPO 简化 | 180 s |
| PPO minimal | 90 s |
| sb3 PPO | 70 s |

→ CartPole 实在太轻量。LLM-RL 会变成"5090 跑一夜"。

---

## Slide 13 · Render 看 agent 表现

```python
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(action)
    if term or trunc: break
env.close()
```

→ 训好的 agent 应能"撑满 500 步"。

---

## Slide 14 · 错误案例 1：reward 卡 50

可能原因：
- lr 过大（5e-3 而非 3e-4）
- adv 未 normalize
- entropy 系数过高，policy 太随机

修复：按 L06 七件套排查。

---

## Slide 15 · 错误案例 2：reward 突然崩塌

可能原因：
- 模式塌缩（entropy → 0 但还没学好）
- 单 batch 出现极端 advantage → 梯度爆炸
- K_epoch 过大，过拟合一个 batch

修复：加 `max_grad_norm=0.5` + `K_epoch ≤ 4`。

---

## Slide 16 · 调参直觉清单

| 改 | 后果 |
|----|------|
| lr ×2 | 更激进，可能崩 |
| ε ×2 | clip 不再起作用 |
| K_epoch ↑ | 数据效率上升但风险增 |
| gamma ↑ | 长期视野，但 advantage 方差大 |
| ent_coef ↑ | 探索增，收敛慢 |

---

## Slide 17 · 进阶任务

学完 CartPole，建议自己跑一遍以下任务（不算正式 capstone）：
- LunarLander-v2 / -v3：状态 8 维，action 4 个，需要更长训练
- Acrobot-v1：sparse reward，PPO 训不动需 reward shaping
- MountainCar-v0：sparse reward + 长 horizon，REINFORCE 完全训不动 → 体验 reward shaping 的必要

---

## Slide 18 · 与 LLM-RL 的衔接

CartPole 经验 → LLM-RL 的转换：

| CartPole | LLM-RL |
|----------|--------|
| obs = 4-dim | obs = token ids（变长） |
| action = 2 | action = vocab_size |
| reward 每步 +1 | reward 仅末端给 |
| ep 长度 ≤ 500 | response 长度 ≤ 256~2k |
| no ref model | + ref model + KL penalty |

L08 开始对接 LLM。

---

## Slide 19 · 自测题

1. 比较 5 算法在 CartPole 上的"100k step reward"。
2. 若 KL approx 突然超过 0.1，最可能的原因？
3. CartPole 上"obs normalization" 为何不需要？
4. `terminated` 与 `truncated` 各自什么时机触发？
5. 写一段 4 行代码，用 sb3 训完后渲染 agent。

---

## Slide 20 · 入口

```bash
# 完整 lab
python learning/rl-foundations/src/cartpole_full.py --algo ppo --total-steps 100_000

# TensorBoard
tensorboard --logdir runs/

# 5 算法横向（约 15 分钟）
for algo in reinforce a2c trpo ppo sb3_ppo; do
    python src/cartpole_full.py --algo $algo --total-steps 50_000
done
```

下一讲：**L08 PPO for LLM** — 把 CartPole PPO 算法搬到 GPT-2 上。
