# L06 · PPO 工程 7 件套

> 20 slides | 50 min | RL Foundations 系列第 6 讲

> 参考：Engstrom et al. 2020 "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

---

## 学习目标

1. 列出 PPO 实现的 7 个关键工程细节
2. 量化每个 trick 对最终性能的贡献（ablation 数据）
3. 看懂 sb3 PPO 源码中的每个 trick 在哪一行

---

## Slide 1 · 为什么"工程细节"重要

Engstrom 2020 的核心发现：**PPO 的性能提升大部分来自"实现细节"而非"clip 算法本身"**。

把 PPO 与 TRPO 的"工程细节"对齐后，性能差距几乎消失。

→ 教训：**永远不要忽视工程细节**。

---

## Slide 2 · 7 件套清单

| # | trick | sb3 对应 | 收益 |
|---|-------|---------|------|
| 1 | advantage normalization | `normalize_advantage=True` | ⭐⭐⭐ |
| 2 | gradient clipping (norm) | `max_grad_norm=0.5` | ⭐⭐ |
| 3 | orthogonal init | sb3 默认 | ⭐⭐ |
| 4 | value loss clipping | `clip_range_vf` | ⭐ |
| 5 | observation normalization | `VecNormalize` | ⭐⭐ |
| 6 | learning rate annealing | `learning_rate=lambda` | ⭐ |
| 7 | reward clipping | `VecNormalize(clip_reward=10)` | ⭐ |

---

## Slide 3 · Trick 1 · Advantage Normalization

```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

**作用**：
- advantage 数量级随任务剧烈变化（CartPole 几十，Mujoco 千级）
- 归一化让 PPO clip 阈值 `ε=0.2` 有"绝对意义"

**消融**：CartPole 中无归一化 reward 跌 40+。

---

## Slide 4 · Trick 1 注意事项

- 归一化只在 **minibatch 之外**做一次（用整个 rollout 的 mean/std）
- 不要再除以 `running mean` —— 那是 sb3 `VecNormalize` 的事
- 不要把 advantage 归到 `[-1, 1]` —— 用 z-score（减均值除标准差）

---

## Slide 5 · Trick 2 · Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**作用**：防 advantage 异常大时梯度爆炸。

**经验值**：`max_norm = 0.5`（PPO 与 sb3 一致）。

**消融**：去掉后偶发训练崩溃，CartPole 表现波动 ±50。

---

## Slide 6 · Trick 3 · Orthogonal Initialization

```python
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.zeros_(layer.bias)
# 输出层用更小 gain：actor 0.01, critic 1.0
```

**作用**：避免初始梯度爆炸；保持 forward variance 不放大。

来自 ICLR 2014 "Exact solutions to the nonlinear dynamics of learning..."

---

## Slide 7 · Trick 3 · 输出层 gain 的细节

| 层 | gain |
|----|------|
| 隐藏层 | √2（ReLU/Tanh 标准） |
| actor 输出层 | 0.01（初始策略近均匀） |
| critic 输出层 | 1.0 |

actor 用 0.01 的原因：让 softmax 初始输出接近均匀，避免一开始就 deterministic。

---

## Slide 8 · Trick 4 · Value Loss Clipping

```python
V_pred_clipped = V_old + (V_pred - V_old).clamp(-eps_vf, eps_vf)
L_vf = max(
    (V_pred - returns)²,
    (V_pred_clipped - returns)²,
).mean()
```

**作用**：与 actor 的 clip 类比，限制 critic 单步变化。

**消融**：CartPole 几乎无影响；Mujoco 显著（5-10pp）。
sb3 默认 `clip_range_vf=None`（不用）。

---

## Slide 9 · Trick 5 · Observation Normalization

```python
env = VecNormalize(env, norm_obs=True, norm_reward=True)
# 实际上：维护 running mean / std 的 obs，z-score 化
```

**作用**：obs 数量级不一致时（cart pos -2~2, ang vel 几百），NN 输入分布不稳。

**消融**：CartPole 无所谓，Mujoco/Atari 关键。

---

## Slide 10 · Trick 6 · Learning Rate Annealing

```python
lr = lambda progress: 3e-4 * (1 - progress)
```

线性衰减到 0。

**作用**：训练后期 fine-tune，避免抖动。

**消融**：CartPole 无所谓；长 budget 训练有效。

---

## Slide 11 · Trick 7 · Reward Clipping

```python
reward = np.clip(reward, -10, 10)
```

**作用**：极端 reward（如 game over 时 -1000）会主导 advantage。

**消融**：CartPole 无（reward 永远 1）；Atari 关键。

---

## Slide 12 · 7 件套消融汇总（CartPole-v1，sb3 PPO baseline 500）

| 关掉的 trick | 200k step 后 mean reward |
|------------|------------------------|
| baseline (全开) | 500 |
| -1 adv norm | 350 |
| -2 grad clip | 460 (波动 ±50) |
| -3 orthog init | 470 |
| -4 vf clip | 500 |
| -5 obs norm | 500 |
| -6 lr anneal | 500 |
| -7 reward clip | 500 |

→ **adv norm 影响最大**，其他在 CartPole 中相对次要。

---

## Slide 13 · 7 件套消融汇总（HalfCheetah，sb3 PPO baseline 3500）

| 关掉的 trick | mean reward |
|------------|------------|
| baseline | 3500 |
| -1 adv norm | 2200 |
| -5 obs norm | 1800 |
| -6 lr anneal | 3000 |
| 全部关 | < 500 |

→ Mujoco 上 **obs norm + adv norm** 联合贡献巨大。

---

## Slide 14 · LLM-RL 中的 7 件套对应

| trick | LLM-PPO 中是否用 |
|-------|----------------|
| adv norm | ✓ 必须 |
| grad clip | ✓ max_norm=1.0 |
| orthog init | ✗（用预训练 LM）|
| vf clip | ✗（critic 是 LM + head）|
| obs norm | ✗（input 是 token，无须归一）|
| lr anneal | ✓ 可选 |
| reward clip | ✓ KL penalty 实质同效 |

---

## Slide 15 · 顺便：KL Penalty（LLM-RL 关键）

LLM-RL 在 PPO loss 上再加一项：

```
L_total = L_clip + c_v · L_vf - β · log(π / π_ref)
```

`π_ref` 是 SFT 后的冻结模型。

**作用**：阻止 policy 离 SFT 太远（防止"reward hacking + 输出胡言乱语"）。

→ L08 详讲。

---

## Slide 16 · 消融实验代码骨架

```python
# 单独关 1 个 trick
configs = [
    dict(name="baseline", **base_cfg),
    dict(name="-adv_norm", adv_norm=False, **base_cfg),
    dict(name="-grad_clip", max_grad_norm=1e6, **base_cfg),
    dict(name="-orthog", orthog_init=False, **base_cfg),
]
for cfg in configs:
    final = train(cfg)
    print(f"{cfg['name']}: {final}")
```

完整版见 `src/ppo_tricks_ablation.py`。

---

## Slide 17 · 调试时按这个清单查

PPO 训不动？按顺序：

1. advantage 归一化？ → 加
2. lr 是否过大？ → 3e-4 → 1e-4 试
3. clip ε 是否合适？ → 0.2
4. K_epoch 太大？ → 减到 4
5. critic loss 数量级合理？ → c_v 调整
6. KL approx 是否爆炸？ → 看输出

---

## Slide 18 · 一句话总结

> PPO 的 clip 算法只是骨架，**adv norm + grad clip + orthog init 三件套**是 PPO 能跑的肉。

---

## Slide 19 · 自测题

1. 为什么 actor 输出层用 gain=0.01？
2. advantage normalization 在 K_epoch × M_minibatch 的什么时机做？
3. value loss clipping 与 actor clip 的本质区别？
4. 在 LLM-RL 中，obs normalization 为何不需要？
5. KL penalty 与 clip 在控制策略变化上的差异？

---

## Slide 20 · 入口

```bash
# 消融实验
python learning/rl-foundations/src/ppo_tricks_ablation.py
# 输出：5 配置的最终 reward
```

下一讲：**L07 CartPole Lab** — gymnasium API、渲染、tensorboard 完整实战。
