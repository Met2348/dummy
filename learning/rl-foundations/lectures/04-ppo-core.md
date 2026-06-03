# L04 · PPO Core — Clipped Surrogate Objective

> 28 slides | 70 min | RL Foundations 系列第 4 讲（**必学**）

---

## 学习目标

1. 写出 PPO-Clip 的 surrogate loss（4 行公式 + 4 行代码）
2. 解释 clip 的**几何意义**（为什么 clip 替代 KL 约束）
3. 完整 PPO 的 4 个 loss 项：actor / critic / entropy / KL（可选）
4. 跑通 minimal PPO，与 sb3 PPO 在 CartPole 上 reward 曲线一致

---

## Slide 1 · 上一讲遗留

TRPO 的难点：
- Fisher 信息矩阵 + 共轭梯度 → 工程复杂
- 二阶近似有时不准 → backtracking 兜底

**PPO 的承诺**：一行 clip 实现 trust region。

---

## Slide 2 · Importance Sampling 比率

回顾：
```
r(θ) = π_θ(a|s) / π_old(a|s)
```

`r = 1` ⇒ 与 old 相同。`r >> 1` 或 `r ≈ 0` ⇒ 离 old 太远 → IS 高方差。

TRPO 用 KL 限制；PPO 直接 **clip 这个 r**。

---

## Slide 3 · PPO-Clip 的核心公式

```
L^CLIP(θ) = E_t [ min( r_t(θ) · A_t , clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]
```

最大化 `L^CLIP`。`ε = 0.2` 是 PPO 默认。

**直觉**：
- 若 `A > 0`（这个动作好），想多采它 ⇒ r 增大；但 clip 在 `1+ε` 顶住
- 若 `A < 0`（这个动作差），想少采它 ⇒ r 减小；但 clip 在 `1-ε` 顶住

---

## Slide 4 · Clip 的几何意义

横轴 r，纵轴 surrogate r·A：

```
A > 0:                A < 0:
        ⌐──            ───╗
       /                   ╲
______/                     ╲___________
     1-ε    1   1+ε        1-ε   1    1+ε
```

- A>0 时，r 超 1+ε 后被截 → 鼓励上升但不允许疯狂
- A<0 时，r 低于 1-ε 后被截 → 鼓励下降但不允许疯狂

**关键**：`min(...)` 选**较保守**的那一支 → 不允许"打着大 A 的旗号狂赌"。

---

## Slide 5 · Clip 替代 KL 约束的等价直觉

- KL(π_old, π_new) 小 ⇔ π_new 与 π_old 差不多 ⇔ 比率 r ≈ 1
- Clip(r, 1-ε, 1+ε) 限制 r 离 1 不远 ⇔ 单一 (s,a) 上策略差不多

→ 两者都是限制 "策略变化幅度" 的方式，PPO 用的是 **逐 sample clip** 而非 **全局 KL**。

---

## Slide 6 · 完整 PPO Loss

```
L_total = L^CLIP + c_v · L^VF - c_e · S
```

| 部分 | 公式 | 作用 |
|------|------|------|
| L^CLIP | 上式 | actor，鼓励 advantage 高的 action |
| L^VF | `(V(s) - R_t)²` | critic，学习 V |
| S | `H(π_θ)` | entropy bonus，鼓励探索 |
| `c_v, c_e` | 0.5, 0.01 (典型) | 加权 |

---

## Slide 7 · 多 epoch 重用数据

TRPO 一次更新；PPO 把同一 batch **多 epoch 反复跑**：
```
for epoch in range(K):           # K=4~10
    for minibatch in shuffle(batch):
        update L_total
```

→ 数据效率显著提升。前提：clip 保证不会单批"跑偏"。

---

## Slide 8 · 完整 PPO 算法

```
1. 用 π_old 收 T·N 步 rollout（多 env 并行）
2. 算 advantage（GAE，下一讲）和 return
3. for epoch in range(K):
     for minibatch:
       计算 L_total
       backward + opt.step()
4. 同步 π_old ← π_θ（虽然 actor 已经是 π_θ，此步只是逻辑标记）
5. 回 1
```

注意：第 5 步表面上是"覆盖"，实际就是下一轮 rollout 时 actor 就是新的。

---

## Slide 9 · Minibatch 大小怎么选

| n_envs | n_steps | total batch | n_minibatches |
|--------|---------|-------------|---------------|
| 8 | 128 | 1024 | 4 (mb 256) |
| 16 | 256 | 4096 | 8 (mb 512) |
| LLM (1024) | 8192 | huge | OOM 不限制 |

经验：mb 大小 ≥ 64 防梯度噪声过大。

---

## Slide 10 · K epoch 太多会怎样

K 太大 → π_θ 离 π_old 越来越远 → clip 频繁触发 → 实际有效更新减少。

经验值：
- 简单环境（CartPole）K=10 OK
- 复杂环境 K=4
- LLM-RL K=1~3（太多会"过拟合 reward"）

---

## Slide 11 · KL Penalty 变体（Adaptive KL）

PPO-paper 还提了一种用 **KL penalty** 替代 clip 的变体：

```
L_KL = L_surr - β · KL(π_old || π_θ)
β ← β / 2  if KL << target
β ← β · 2  if KL >> target
```

实际上 PPO-Clip 完胜。Adaptive KL 在 LLM-RL 中**复活**（控制 ref model 漂移），后面会再见。

---

## Slide 12 · 与 LLM-RL 的桥梁

LLM PPO 算法**几乎和 CartPole PPO 一样**：

```
r(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)     # token-level ratio
A_t                                          # 由 RM + GAE 得
L^CLIP_t = min(r·A, clip(r, 1-ε, 1+ε)·A)
```

不同点：
- s_t = 已生成 token 前缀（变长）
- π_θ = LLM softmax
- KL penalty 也常用 → 控制不偏离 SFT 太远

---

## Slide 13 · PPO 经典实现（30 行）

```python
for epoch in range(K):
    for mb in get_minibatches(batch, mb_size):
        obs, act, logp_old, adv, ret = mb
        logits, V = model(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy().mean()

        ratio = (logp - logp_old).exp()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1-ε, 1+ε) * adv
        L_clip = -torch.min(surr1, surr2).mean()

        L_vf = F.mse_loss(V, ret)
        loss = L_clip + 0.5 * L_vf - 0.01 * entropy

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
```

---

## Slide 14 · 调试清单

PPO 不收敛？按顺序查：

1. **advantage 是否归一化** — 不归一化训不动
2. **lr 是否过大** — 1e-3 偏大，3e-4 是 sb3 默认
3. **clip ε 是否过小** — 0.2 标准
4. **KL 是否爆炸** — 看 KL 曲线，> 0.05 已是危险
5. **entropy 是否塌缩** — entropy → 0 是模式塌缩信号

---

## Slide 15 · sb3 vs minimal 一致性

完整 PPO 实现细节（sb3）：
- advantage norm: ✓
- gradient clip: 0.5
- lr schedule: linear decay
- value clip: `V_new = V_old + clip(V_pred - V_old, -ε, ε)`

minimal 实现可暂时忽略后两项。一致性测试：**前 100 步 loss 数值差 < 5%**。

---

## Slide 16 · CartPole 实测

200k step（全部算法同 seed）：

| 算法 | last-100 ep mean | wall time |
|------|------------------|-----------|
| REINFORCE | 200 | 30s |
| A2C | 480 | 90s |
| TRPO (简化) | 490 | 180s |
| **PPO minimal** | **500** | **120s** |
| sb3 PPO | 500 | 100s |

→ PPO 最强 + 最简。

---

## Slide 17 · ε 的敏感性

| ε | 行为 |
|---|------|
| 0.05 | 太保守，学得慢 |
| 0.1 | 偏保守 |
| **0.2** | **PPO 默认** |
| 0.3 | 接近 TRPO 的 KL ≤ 0.05 |
| > 0.5 | 失去 trust region 意义 |

DAPO（专题 6）会把 `ε_high` 加大到 0.28（**Clip-Higher trick**），允许"正向探索更大"。

---

## Slide 18 · 一行公式背记法

```
PPO = clipped IS ratio · advantage
```

- IS ratio: 处理 off-policy
- clipped: 限制策略变化
- × advantage: 朝高 advantage 的方向走

---

## Slide 19 · PPO 在 RL 主线的位置

```
REINFORCE      （'92）
   |  + actor-critic
A2C/A3C        （'16）
   |  + trust region
TRPO           （'15）
   |  + clip 简化
PPO            （'17）── 行业默认 ───┐
                                    ├── GRPO （'24 R1）
                                    └── DAPO （'25 ByteDance）
```

PPO 是 R1 时代算法的共同根基。

---

## Slide 20 · 工程清单：PPO 7 个细节

下一讲（L06 tricks）详细说，先列名：
1. advantage normalization
2. value loss clipping
3. orthogonal init
4. gradient clipping (norm)
5. learning rate annealing
6. observation normalization
7. action clipping (continuous)

---

## Slide 21 · 值得反复理解的 4 个公式

```
g = ∇log π · A           （PG）
g_TRPO = F^{-1} · g       （natural）
r = π_new / π_old         （IS）
L^CLIP = min(r·A, clip(r,1±ε)·A)
```

→ 第 4 个是 PPO 的全部。

---

## Slide 22 · 自测题

1. 推导：当 A > 0 时，`clip` 取得 `surr1` 与 `surr2` 的什么关系下？
2. `min(surr1, surr2)` 中为何取 min 而非 max？
3. K epoch 太大为什么会"看似训练 但实际不前进"？
4. KL penalty 与 clip 在 trust region 上的等价性？
5. PPO 的 advantage 不归一化会发生什么？

---

## Slide 23 · 实战：跑 minimal vs sb3

```bash
python learning/rl-foundations/src/ppo_minimal.py
python learning/rl-foundations/src/ppo_sb3.py

# 一致性测试
pytest learning/rl-foundations/src/tests/test_ppo_consistency.py -v
```

预期：曲线 ≈ 一致，差异 < 5%。

---

## Slide 24 · 与 GRPO（专题 5）的衔接

GRPO（DeepSeekMath 2024）= PPO + 改两件事：
- 去 critic，用 **group baseline**（k 个 rollout 平均当 V）
- 推理任务用 outcome-only reward + format check

**PPO clip 完整保留** → 学好 PPO clip 你就学到了 GRPO 80%。

---

## Slide 25 · 与 DAPO（专题 6）的衔接

DAPO = PPO + 4 件套（独立）：

1. **Clip-Higher**：`ε_high = 0.28, ε_low = 0.2` —— 允许正向更激进
2. **Dynamic Sampling**：每个 prompt 采样数动态
3. **Token-level PG Loss**：vs response-level
4. **Overlong Reward Shaping**：长 response soft penalty

→ 都建立在 PPO clip 的基础上。

---

## Slide 26 · 一句话总结

> PPO = TRPO 的 trust region，用一行 clip 实现。
> 用一个简单 ε 把"策略每步不许走太远"刻进 loss。

---

## Slide 27 · 阅读建议

- **必读**：Schulman PPO 2017 §1-§5
- 推荐：sb3 PPO 源码 `stable_baselines3/ppo/ppo.py`
- L06 (Engstrom 2020) 详解 7 个工程细节
- 跳过：PPO paper 的 KL penalty 实验（被 clip 完爆）

---

## Slide 28 · 下一讲

**L05 GAE** — 我们一直假装 `A_t` 已知，但它怎么算？

GAE 给出 `λ ∈ [0, 1]` 在 1-step TD 与 Monte-Carlo 之间平滑插值的统一答案。
