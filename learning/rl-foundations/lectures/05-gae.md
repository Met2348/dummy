# L05 · Generalized Advantage Estimation (GAE)

> 22 slides | 55 min | RL Foundations 系列第 5 讲

---

## 学习目标

1. 理解 1-step TD 与 Monte-Carlo advantage 的 bias-variance trade-off
2. 推导 GAE 公式 `A^GAE_t = Σ (γλ)^l δ_{t+l}`
3. 写出 GAE 反向计算的 5 行代码
4. 理解 λ 在 PPO 中怎么选

---

## Slide 1 · 上一讲遗留

PPO 公式中：

```
L^CLIP = E [ min(r·A, clip(r,1±ε)·A) ]
```

**A 怎么算？** 上一讲简单说"1-step TD 或 MC"，本讲给出统一答案。

---

## Slide 2 · n-step advantage

定义 TD 误差：
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

n-step advantage：
```
A^(n)_t = δ_t + γ δ_{t+1} + γ² δ_{t+2} + ... + γ^{n-1} δ_{t+n-1}
       = (Σ_{l=0..n-1} γ^l r_{t+l}) + γ^n V(s_{t+n}) - V(s_t)
```

- `n=1`: 1-step TD（bias 高，variance 低）
- `n=∞`: Monte-Carlo（bias 低，variance 高）

---

## Slide 3 · n 的两难选择

| n 小 | n 大 |
|------|------|
| bias 高（V 错就传递误差） | bias 低（真实 reward 主导） |
| variance 低 | variance 高 |
| 快速学习但局部错 | 慢但正确 |

→ **GAE 把所有 n 加权平均**。

---

## Slide 4 · GAE 公式

```
A^GAE_t(λ) = Σ_{l=0..∞} (γλ)^l δ_{t+l}
```

`λ ∈ [0, 1]` 是新超参。

| λ | 等价 |
|---|------|
| 0 | A = δ_t (1-step TD) |
| 0.5 | 50/50 |
| 0.95 | PPO 默认 |
| 1 | A = G_t - V(s_t)（MC） |

---

## Slide 5 · GAE 的递推关系

直接算 `Σ (γλ)^l δ_{t+l}` 要看到未来。

**逆序递推**：
```
A^GAE_t = δ_t + γ λ A^GAE_{t+1}
```

→ 从 t=T 开始逆序累加。

---

## Slide 6 · GAE 的递推代码（5 行）

```python
T = len(rewards)
A = torch.zeros(T)
gae = 0.0
for t in reversed(range(T)):
    mask = 1.0 - dones[t]                       # done 时不传递
    delta = rewards[t] + gamma * V[t+1] * mask - V[t]
    gae = delta + gamma * lam * mask * gae
    A[t] = gae
returns = A + V                                 # 给 critic 训练
```

注意 `mask` —— done 处把传递关系切断。

---

## Slide 7 · 推导：Σ (γλ)^l δ_{t+l}

```
A^GAE_t = Σ_{l=0..∞} (γλ)^l δ_{t+l}
       = δ_t + γλ · Σ_{l=0..∞} (γλ)^l δ_{t+1+l}
       = δ_t + γλ · A^GAE_{t+1}   ✓
```

→ 与上 slide 递推一致。

---

## Slide 8 · 推导：λ=1 时 GAE 等于 MC

```
A^GAE_t(λ=1) = Σ_l γ^l δ_{t+l}
            = Σ_l γ^l [r_{t+l} + γ V(s_{t+l+1}) - V(s_{t+l})]
            = (Σ_l γ^l r_{t+l}) + (telescoping V 项相消)
            = G_t - V(s_t)
```

中间 V 项前后相消（telescoping sum），优美。

---

## Slide 9 · 推导：λ=0 时 GAE 等于 1-step TD

```
A^GAE_t(λ=0) = δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

直接退化。

---

## Slide 10 · λ 怎么选

经验值：

| 任务 | λ |
|------|---|
| CartPole | 0.95 |
| Atari | 0.95 |
| MuJoCo | 0.95 |
| **PPO 默认** | **0.95** |
| LLM-RL | 0.95-1.0 |
| R1-Zero (GRPO) | 1.0（实际上 GRPO 没 critic，等价 MC）|

→ 0.95 几乎是"万能"。

---

## Slide 11 · GAE 的 returns 怎么用

GAE 给出 advantage `A`，那 critic 的训练 target 是什么？

**A + V**：
```
returns[t] = A^GAE_t + V(s_t)
critic_loss = (V_predict(s_t) - returns[t])²
```

→ critic target 是 "GAE-improved estimate of return"。

---

## Slide 12 · 直观对比图（待自己画）

横轴 t，纵轴 advantage：
- 蓝色 1-step TD（抖）
- 橙色 MC（更"尖锐"，看到未来 reward 才知道）
- 绿色 GAE λ=0.95（平滑过渡）

→ GAE 在 t 早期更接近 TD（噪声小），在 t 后期更接近 MC（看 future 准）。

---

## Slide 13 · GAE 与 critic 的相互依赖

GAE 公式里有 `V`，但 `V` 又要靠 GAE return 训。

实际：
1. critic forward 一次拿 `V`
2. 算 GAE 拿 `A` 和 `returns`
3. critic 用 `returns` 当 target 更新
4. 下次 forward 用更新过的 `V`

→ 这是 **fixed-point iteration**，PPO 训练过程其实在求 critic 的 fixed point。

---

## Slide 14 · Bootstrapping vs MC：偏差来源

- 1-step TD 偏差来自 **不准的 V**
- MC 没有偏差（用真实 reward）但方差大

GAE λ 调节"信任 V"的程度：
- λ 小 → 信 V → bias V 的错误
- λ 大 → 不信 V，用真实 reward → variance 高

---

## Slide 15 · GAE 在 PPO 中的实际计算时机

```
1. rollout T 步，拿 obs/act/rew/done 和 V
2. bootstrap: 算 last_V = V(s_{T})
3. 反向 GAE：上面 5 行代码
4. advantage normalization: (A - μ) / σ
5. flatten 到 (T·N, ...) shape
6. K epoch × M minibatch 更新
```

注意 step 4 normalization 在 GAE 之后做（不是之前）。

---

## Slide 16 · LLM-RL 中的 GAE

每个 (prompt, response) 一条 episode：
- s_0 = prompt
- a_t = token_t
- s_{t+1} = prompt + token_0...token_t
- r_t = 通常仅在最后给（RM 打分）
- T = response 长度

→ 中间步 `r_t = 0`，只有最后非零 → GAE 把 RM 打分均匀"广播"回每个 token。

---

## Slide 17 · LLM-RL 中 GAE 的别名：reward 归因

末端 reward 1.0 → 经过 GAE 反向（γ=1, λ=1）→ 每个 token 的 advantage ≈ A_final - V_pred。

这等价于：**用 critic 来归因 reward 到每个 token**。

→ critic 的好坏直接决定 token-level advantage 是否准确。

---

## Slide 18 · 工程坑

1. **last_V 漏算** → 末端 advantage 偏差
2. **done mask 漏** → episode 边界传递 → 偏差
3. **adv 不归一化** → loss 数量级飘
4. **K_epoch 过多** → GAE 计算的 advantage 与新 policy 不匹配（过时）

---

## Slide 19 · 数值小练习

给定 3 步：
```
rewards = [1, 1, 1]
values  = [0.5, 0.3, 0.2]
dones   = [F, F, T]
gamma=1, lam=0.95
last_V = 0
```

计算 GAE：

```
t=2: delta = 1 + 1·0·(1-1) - 0.2 = 0.8; A_2 = 0.8
t=1: delta = 1 + 1·0.2·(1-0) - 0.3 = 0.9; A_1 = 0.9 + 1·0.95·1·0.8 = 1.66
t=0: delta = 1 + 1·0.3·(1-0) - 0.5 = 0.8; A_0 = 0.8 + 1·0.95·1·1.66 = 2.377
```

→ 用 `gae.py` 验证：`compute_gae([1,1,1], [0.5,0.3,0.2], [0,0,1], 0, 1.0, 0.95)`

---

## Slide 20 · 自测题

1. λ=0 和 λ=1 各自等价于什么经典方法？
2. 为什么 `A^GAE_t = δ_t + γλ A^GAE_{t+1}`？逐项展开证明。
3. critic 的 training target 是 `A + V` 还是 `A`？为什么？
4. 在 done 边界 mask 漏掉一处会发生什么？
5. λ=1 时 GAE 是无偏的还是无方差的？给数学论证。

---

## Slide 21 · 入口

```bash
# GAE 单元测试
python -m pytest learning/rl-foundations/src/tests/test_ppo_consistency.py::test_gae_numerical -v

# 完整 PPO 跑
python learning/rl-foundations/src/ppo_minimal.py --lam 0.95
python learning/rl-foundations/src/ppo_minimal.py --lam 1.0   # MC
python learning/rl-foundations/src/ppo_minimal.py --lam 0.0   # 1-step TD
# 三组对比 reward 曲线
```

---

## Slide 22 · 一句话总结

> GAE = TD 与 MC 之间的几何加权平均，用 λ 调节信"V"还是信"r"。
> 5 行代码、PPO 默认 0.95、LLM-RL 用 1.0。

下一讲：**L06 PPO 工程 7 件套**。
