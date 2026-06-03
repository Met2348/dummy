# L03 · TRPO：Trust Region Policy Optimization

> 28 slides | 70 min | RL Foundations 系列第 3 讲

---

## 学习目标

1. 理解"PG 步子太大就崩"的本质
2. 推导 **surrogate objective** + 公认的 KL trust-region 约束
3. 看懂 natural gradient 与 Fisher information matrix 是怎么进来的
4. 跑通简化版 TRPO（用 backtracking line search 代替共轭梯度）

---

## Slide 1 · 上一讲遗留：A2C 为何不稳

A2C 训得稳了不少，但仍然脆弱：
- lr 略大 → 训不动
- 一次 update 跨度太大 → 策略一头栽

**原因**：PG 的"局部线性"假设只在小范围成立。

→ **思路**：每次更新前限制策略 **变化幅度**。

---

## Slide 2 · 步子怎么"太大"

直觉：
- old policy：在状态 s 下 80% 选 a₁
- new policy（更新一步后）：90% 选 a₂

这种剧烈翻转后，**采集到的数据来自 old policy，已不适用 new policy**。
→ off-policy 偏差 → bad updates → 崩

---

## Slide 3 · Surrogate Objective

TRPO 优化目标（Importance Sampling 视角）：

```
L_surr(θ) = E_{s,a ~ π_old} [ π_θ(a|s) / π_old(a|s) · A^{π_old}(s, a) ]
```

含义：用 old policy 的样本估 new policy 的预期 advantage。

**关键比率**：

```
r(θ) = π_θ(a|s) / π_old(a|s)
```

`r=1` ⇒ 与 old policy 相同。`r > 1` ⇒ 更倾向选这个 a。

---

## Slide 4 · 为什么 surrogate 不直接用？

因为 `L_surr` 在 `θ` 离 `θ_old` 远时不再可信（importance sampling 高方差）。

→ TRPO 加一个 **trust region** 约束：

```
maximize L_surr(θ)
s.t.    E_s [ KL( π_θ_old(·|s) || π_θ(·|s) ) ] ≤ δ
```

`δ ~ 0.01`：每步策略与 old 的 KL 差不超过这个值。

---

## Slide 5 · 数学：约束优化转 Lagrangian

```
L(θ, λ) = L_surr(θ) - λ · (E[KL] - δ)
```

KKT 条件 → 求解：

```
∇_θ L_surr = λ · ∇_θ E[KL]
```

但是 `∇ E[KL]` 不是简单梯度 → 引出 **Fisher Information Matrix**。

---

## Slide 6 · KL 的二阶展开

在 `θ_old` 附近，

```
E[KL(π_old || π_θ)] ≈ ½ (θ - θ_old)^T F(θ_old) (θ - θ_old)
```

其中 `F = E[∇log π · ∇log π^T]` 是 **Fisher 信息矩阵**。

KL 在 `θ = θ_old` 时为 0；其曲率（Hessian）正是 `F`。

---

## Slide 7 · Natural Gradient

把上一行的 KL 约束代入 Lagrangian，得：

```
θ_new ≈ θ_old + α · F^{-1} · g
```

其中 `g = ∇_θ L_surr | θ_old` 是普通 PG 梯度。

**关键**：`F^{-1} g` 叫 **natural gradient** —— 它把欧氏空间梯度变换成"分布空间"中的最速下降方向。

---

## Slide 8 · Fisher 太大怎么办

`F` 维度 = `(参数数)²`，对深度 NN 完全不可存（GPT-2-medium 124M² ≈ 15 petabytes）。

**TRPO 的解决**：
- 不算 `F^{-1}`
- 用 **共轭梯度 CG** 求解 `F x = g` —— 只需要 `F · v` 的乘积
- `F · v` 可用二次反向传播算（实际上比 GMV 快）

---

## Slide 9 · 完整 TRPO 算法

```
1. 用 π_old 采 N 步数据
2. 算 advantage A^{π_old}
3. 算 PG 梯度 g
4. 共轭梯度求 x = F^{-1} g  (10 步左右)
5. 计算最大步长 α* = √(2δ / x^T F x)
6. θ_new = θ_old + α* · x
7. 验收：算新策略的 L_surr 和实际 KL
8. 若不满足约束 / L_surr 下降 → 回退 backtracking line search
9. 重复
```

---

## Slide 10 · TRPO 之后是 PPO

TRPO 工程复杂：CG、F·v、backtracking line search → 实现易错。

**PPO（下一讲）**：用 **clip ratio** 替代 KL 约束 —— 一行代码完成 "trust region"，工程简洁。

PPO 几乎 100% 取代 TRPO 成为默认 baseline。

---

## Slide 11 · 简化版 TRPO（教学用）

跳过 CG / F·v，直接：
1. 算 `g = ∇ L_surr`
2. **试步**：`θ_try = θ_old + α g`
3. 验收：算 `L_surr(θ_try)` 和 `KL(π_old || π_try)`
4. 若 KL > δ 或 L_surr 下降 → `α /= 2`，重试（**backtracking line search**）
5. 否则 accept

→ 行为类似 TRPO，但少了 natural gradient 的"分布几何"加速。

---

## Slide 12 · CartPole 上 TRPO 表现

(典型数据)

| 算法 | 200k step eval | 评论 |
|------|---------------|------|
| REINFORCE | 200 | 高方差 |
| A2C | 480 | TD 加速 |
| **TRPO** (简化版) | 490 | 单步稳，但慢 |
| PPO | 500 | 几乎完美，下一讲 |

注：TRPO 的 wall time 比 A2C 慢 2× 左右（CG + line search 开销）。

---

## Slide 13 · 关键超参 δ

| δ | 行为 |
|---|------|
| 0.001 | 过于保守，学得慢 |
| 0.01 | **TRPO 经典** |
| 0.05 | 偏激进，可能崩 |
| > 0.1 | KL 太大，丢失 trust region 意义 |

---

## Slide 14 · TRPO 与 SL 的对比

| | SL | TRPO |
|---|----|------|
| 单步目标 | 减 loss | 在 trust region 内最大化 surrogate |
| 约束 | 无 | KL ≤ δ |
| 重复使用数据 | 多 epoch 反复跑 | 一次性更新（PPO 改进） |

---

## Slide 15 · Importance Sampling 高方差

`r(θ) = π_θ / π_old` 当差异大时变得很尖（一些 r >> 1，另一些 ≈ 0），方差爆炸。

TRPO 用 KL 约束限制 r 的偏离 → 间接控制 IS 方差。

PPO 用 **直接 clip r 到 [1-ε, 1+ε]** 取代这一约束。

---

## Slide 16 · Fisher Information Matrix 的另一视角

| 视角 | 含义 |
|------|------|
| 信息几何 | 概率流形的 Riemannian metric |
| 二阶优化 | NLL 的 Hessian 的期望 |
| KL 曲率 | KL(π_old, π_θ) 在 θ_old 附近的 Hessian |

→ 在 RL 之前，自然梯度已在 SL 出现（Amari 1998）。

---

## Slide 17 · 共轭梯度的迭代过程（直观）

CG 解 `Ax = b`，从 `x_0 = 0` 开始：
- 每步沿当前方向最小化
- 下次方向与之前方向**共轭**（不重复）
- N 维问题 N 步内精确解，深 NN 通常 10 步够好

类比：在 N 维迷宫里走，每步不沿曾走过的方向。

---

## Slide 18 · F · v 怎么算

`F = E[∇log π · ∇log π^T]`，直接展开矩阵不可行。

**Pearlmutter trick**：
```
F · v = ∇_θ ( (∇_θ KL)^T · v )
```

即 **算一次 KL 梯度，再对它做向量乘 + 二次反向**。这是 Hessian-vector product 在 RL 里的版本。

实操：`torch.autograd.grad(kl_grad @ v, model.parameters())`。

---

## Slide 19 · 工程坑 1：F 接近 singular

观察：CG 不收敛、α* 极大、KL 严重违反。

原因：F 接近 singular（数值上 ill-conditioned）。

修复：加阻尼 `F + λI`（λ ~ 0.1），叫 **damped Fisher**。

---

## Slide 20 · 工程坑 2：实际 KL >> δ

backtracking line search 没保护到位时：
- 算出来的 `α*` 用上去后实际 KL 是 0.05（远超 0.01）

原因：二阶近似在大步时失效。

修复：
- accept 前实测 KL，不满足就 `α /= 2`
- 限制最大 backtracks（典型 10）

---

## Slide 21 · TRPO 在 LLM 上的角色

历史价值：第一次让 PG 在大网络上稳定 → 启发后续。

实际很少用：
- PPO 简单 10x，性能差不多
- LLM 4 模型 + Fisher 一起是噩梦

但 **R1-Zero 的 GRPO 算法**间接继承了 TRPO 的"约束" idea（用 group baseline 替代 critic + KL penalty）。

---

## Slide 22 · 阅读建议

- **必读**：Schulman et al. 2015 §1-§4
- **跳过**：附录 D 的 vine sampling 与 fixed-point KL（实操几乎不用）
- **参考**：spinningup TRPO 实现 + Schulman PhD 论文 §3

---

## Slide 23 · 简化 TRPO 实现要点

代码骨架（详见 `src/trpo_minimal.py`）：

```python
g = compute_pg_gradient(policy, batch, advantages)
# 简化：跳过 CG，直接当成 natural gradient ≈ g
direction = g
step = 0.01
for _ in range(10):    # backtracking
    set_params(theta_old + step * direction)
    kl = compute_kl(old_policy, new_policy)
    if kl > 0.01 or L_surr_new < L_surr_old:
        step /= 2
    else:
        break  # accept
```

---

## Slide 24 · 与 PPO 的桥梁

下一讲的核心 trick：

```
L_clip = E [ min( r·A, clip(r, 1-ε, 1+ε)·A ) ]
```

PPO 用 clip 替代 TRPO 的 KL 约束 —— 不再需要 F、CG、KL 测量。

→ 简单、稳、deep RL 的事实标准。

---

## Slide 25 · 自测题

1. 为什么 `θ_new ≈ θ_old + α F^{-1} g` 比 `θ_new = θ_old + α g` 更"自然"？
2. F 是什么，为什么不直接存？
3. 在简化版 TRPO 里，backtracking line search 实际起到什么作用？
4. 描述 `r(θ) = π_θ / π_old` 在采样分布上的"漂移"问题。
5. 若 KL 超约束，TRPO 应该做什么？

---

## Slide 26 · 历史与影响

- 2015 Schulman TRPO：让 PG 第一次在 deep policy NN 上稳
- 2017 Schulman PPO：极简化 → 工业默认
- 2024 GRPO（DeepSeekMath）：去 critic + group baseline，但保留 PPO 的 clip
- 2025 DAPO：在 PPO 基础上加 Clip-Higher，反向上溯到 TRPO 思路

→ trust-region 思路贯穿了整个 deep RL。

---

## Slide 27 · 入口

```bash
python learning/rl-foundations/src/trpo_minimal.py
# 预期：50k step CartPole eval mean ≥ 400
```

下一讲：**L04 PPO — clip 是怎么"工程地"实现 trust region 的**。

---

## Slide 28 · 一句话总结

> TRPO = 用 KL 约束保证每次策略更新在"可信"范围内。
> PPO = 用 clip ratio 把这件事变成一行代码。
