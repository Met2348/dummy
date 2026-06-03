# L02 · GRPO 完整推导 — Group Relative Policy Optimization

> 28 slides | 70 min | Reasoning R1 第 2 讲 ⭐⭐⭐⭐⭐ 必学

> DeepSeekMath 2024.02 提出 / R1 的算法核心

---

## 学习目标

1. 推导 GRPO loss（从 PPO 出发，去 critic，用 group baseline）
2. 理解为什么 GRPO 适合推理任务（无需 V）
3. 写出 GRPO 的 25 行 Pytorch 实现
4. 与 PPO 在 advantage 估计上的本质差异

---

## Slide 1 · 起点 · PPO 回顾

```
L_PPO = -E [ min(r·A, clip(r,1±ε)·A) ]
A_t = GAE(λ=0.95)            # 需要 critic V(s)
```

**问题**：critic 是另一个 LLM，巨大 / 训练慢 / 不稳。

→ 能否**去掉 critic**？

---

## Slide 2 · 去 critic 的两种思路

| 思路 | 例 | 缺 |
|------|-----|----|
| Monte-Carlo return | REINFORCE | 高方差 |
| **Group baseline** | **GRPO** | 需多次 rollout |

GRPO：**每个 prompt 多采样 k 条 response，用它们的 reward 均值当 baseline**。

---

## Slide 3 · Group Baseline 定义

对 prompt `x`，采 k 条 response `y_1, ..., y_k`，得 reward `R_1, ..., R_k`。

**Group mean**: `R̄ = (R_1 + ... + R_k) / k`
**Group std**: `σ = std(R_1, ..., R_k)`

**Advantage**：
```
A_i = (R_i - R̄) / σ
```

→ 第 i 条 response 相对组内平均有多好（z-score 化）。

---

## Slide 4 · GRPO loss 公式

```
L_GRPO = -E_i [ Σ_t min( r_i,t · A_i , clip(r_i,t, 1±ε) · A_i ) - β · KL(π_θ || π_ref) ]
```

其中：
- `r_i,t = π_θ(y_i,t | s_t) / π_old(y_i,t | s_t)` —— per-token ratio
- `A_i` 是 response-level（对该 response 所有 token 一样）
- KL penalty 作为 loss 项加（**不是加在 reward 上**）

---

## Slide 5 · 与 PPO 的对比

| | PPO | GRPO |
|---|----|------|
| critic | ✓ V network | ✗ |
| advantage | GAE token-level | group-z-score response-level |
| KL | reward 内 / loss 外都行 | loss 外 |
| 模型数 | 4 (actor + critic + ref + RM) | 3 (actor + ref + RM/rule) |
| 显存 | 4× | 3× |

---

## Slide 6 · 为什么 group baseline 在推理任务好

推理任务的 reward 特性：
- 仅末端给（success/failure）
- 稀疏（多数 rollout 是失败）
- 准/不准的离散判断

**Group**：8 条 response 中 2 条对 6 条错 → `A_correct = +0.7, A_wrong = -0.3` → 信号清晰。

**PPO + GAE**：需要 critic 学到 V(s)，但稀疏 reward 下 critic 几乎学不动。

---

## Slide 7 · 采样数 k 怎么选

| k | 行为 |
|---|------|
| 4 | 最小，advantage 估计 noisy |
| **8** | **R1 / TinyZero 默认** |
| 16 | 更稳但显存 / 时间 × 2 |
| > 32 | 边际收益递减 |

显存约束：rollout 长度 256，k=8 →  batch 8 × 256 = 2048 token 一次。

---

## Slide 8 · GRPO 训练 loop

```
1. 收 prompts batch
2. 对每个 prompt 采 k 条 response (用 vllm 加速)
3. 算 reward（rule-based for R1：format + accuracy）
4. 算 group advantage A_i = (R_i - R̄) / σ
5. 计算 per-token log_probs (新旧 actor + ref)
6. L_GRPO = clip surrogate + β · KL(π_θ || π_ref)
7. backward + opt.step
```

---

## Slide 9 · KL penalty 在 loss 外加

与 PPO 不同，GRPO 把 KL **加到 loss**（而非 reward）：

```
L_total = L_clip - β · KL(π_θ || π_ref)
KL = E_t [ KL( π_θ(.|s_t) || π_ref(.|s_t) ) ]
```

**优势**：
- KL 在 loss 中梯度更直接
- 不需要 token-level reward adjustment
- 调参更稳定

---

## Slide 10 · KL 的 unbiased estimator

GRPO 用 Schulman 的 unbiased KL estimator：

```
KL(π_θ || π_ref) ≈ E[ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1 ]
```

vs simple KL = -log(π_ref/π_θ) — 后者 unbiased 但高方差。

---

## Slide 11 · 完整 GRPO loss 公式

```
L_GRPO = -E [
    Σ_t in response  min( r_t · A , clip(r_t, 1±ε) · A )
                   - β · KL_t
]
```

其中：
- A = group-z-score advantage（response-level，对所有 t 一样）
- r_t = log π_θ(y_t|s_t) - log π_old(y_t|s_t)，exp 后是 ratio
- KL_t = exp(log_ref - log_θ) - (log_ref - log_θ) - 1

→ 一行公式覆盖 GRPO 全部。

---

## Slide 12 · 与 RLOO 的关系

RLOO (Leave-One-Out 2024)：

```
A_i = R_i - (Σ_{j≠i} R_j) / (k-1)   # 用其他 k-1 个的均值当 baseline
```

GRPO 用全部 k 个的均值当 baseline，**包括自己**。

实证：差异很小，RLOO 略 unbiased 一点点。

---

## Slide 13 · GRPO 在 R1-Zero 的应用

R1-Zero (DeepSeek 2025.01) 用 GRPO + rule reward + NO SFT cold-start：
- 基座 DeepSeek-Math-7B
- 训练 8k step
- 涌现"aha moment"

我们的 capstone Track A 用 GPT-2-medium + Countdown-3 复现（教学规模）。

---

## Slide 14 · 25 行 Pytorch 实现（核心）

```python
# 1. rollout k responses per prompt
responses, log_probs_old, rewards = rollout_k(actor, prompts, k=8)

# 2. group advantage
rewards = rewards.reshape(B, k)         # (B, k)
A = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-8)
A = A.flatten()                          # (B*k,) — 每条 response 一个值

# 3. K epoch update
for epoch in range(K):
    for mb in get_minibatches():
        log_probs_new = get_log_probs(actor, mb)
        log_probs_ref = get_log_probs(ref, mb)

        ratio = (log_probs_new - log_probs_old).exp()    # per-token
        surr1 = ratio * A.unsqueeze(1)                    # broadcast to all tokens
        surr2 = ratio.clamp(1-eps, 1+eps) * A.unsqueeze(1)
        L_clip = -torch.min(surr1, surr2).mean()

        kl_t = (log_probs_ref - log_probs_new).exp() - (log_probs_ref - log_probs_new) - 1
        L_total = L_clip + beta * kl_t.mean()

        L_total.backward(); opt.step()
```

---

## Slide 15 · 一致性测试与 PPO

`tests/test_grpo_consistency.py`：
- 当 k=1 时，GRPO advantage = 0 → 不更新（边界情况）
- 当 critic V ≡ R̄ 时，GRPO 等价 PPO（理论）
- minimal vs verl 实现 loss < 1e-6

---

## Slide 16 · 训练超参（R1-Zero typical）

| 超参 | 值 |
|------|---|
| lr | 1e-6 |
| batch (prompts) | 32-64 |
| k (group) | 8 |
| max_response_len | 1024 (R1) / 256 (toy) |
| ε (clip) | 0.2 |
| β (KL) | 0.001-0.01 |
| K_epochs | 1-2 |

---

## Slide 17 · Reward 设计：rule-based

R1-Zero 的 reward：
```
r = α · r_format + (1-α) · r_accuracy

r_format = 1 if response matches <think>...</think><answer>...</answer> else 0
r_accuracy = 1 if extract_answer(response) == ground_truth else 0
α = 0.1 (format)
```

→ 完全 rule-based，**无需 RM**。

---

## Slide 18 · 与 RL Foundations PPO 的桥梁

GRPO = PPO 的"轻量推理版"：
- 去 critic → 节约 1/4 显存
- 用 group baseline → 推理稀疏 reward 下更好
- 用 rule reward → 完全去 RM

→ 学好 PPO 你就理解了 GRPO 的全部。

---

## Slide 19 · 训练时长（典型）

- GPT-2-medium + Countdown + GRPO k=8：100 step ~ 1h（5090）
- Qwen-1.5B + LoRA + GSM8K + GRPO k=8：100 step ~ 4h（5090）
- DeepSeek-Math-7B + GRPO + DAPO：8000 step ~ 1 周（多卡）

---

## Slide 20 · 与 DAPO 的衔接

DAPO (2025.03) = GRPO + 4 件套：
1. Clip-Higher：ε_high=0.28，ε_low=0.2（不对称 clip）
2. Dynamic Sampling：每 prompt 采样数动态
3. Token-level PG Loss：vs response-level
4. Overlong Reward Shaping：长 response 软惩罚

→ 专题 6 详讲。GRPO 是 DAPO 的基础。

---

## Slide 21 · 与 PRIME 的关系

PRIME 在 GRPO 基础上加 **隐式 PRM**：
- 从 outcome reward 自动学步级 reward
- 训练 actor 同时训练 implicit PRM head

→ 专题 6 / 专题 4 详讲。

---

## Slide 22 · 自测题

1. GRPO 与 PPO 在 advantage 估计上的本质差异？
2. Group baseline 在稀疏 reward 下为何比 critic V 好？
3. 推导 GRPO 的 KL 项为何用 `exp(δ) - δ - 1` 而非 `-δ`？
4. k=8 与 k=4 的 trade-off？
5. GRPO 的 rule reward 适用什么任务，不适用什么？

---

## Slide 23 · 工程注意

| 项 | 注意 |
|----|------|
| rollout | 用 vllm，加速 10× |
| advantage | 在 group 内 z-score |
| KL | unbiased estimator + 在 loss 外 |
| K_epoch | 1-2 即可，不要太大 |
| lr | 1e-6 起步 |

---

## Slide 24 · 阅读建议

- **必读**：DeepSeekMath GRPO §3
- **必读**：DeepSeek-R1 §2
- **推荐**：verl 的 GRPO 实现源码
- **可选**：RLOO (Ahmadian 2024) 对照

---

## Slide 25 · 入口

```bash
# WSL2 中跑
python learning/reasoning-r1/src/grpo_minimal.py
python learning/reasoning-r1/src/grpo_verl.py
pytest learning/reasoning-r1/src/tests/test_grpo_consistency.py
```

---

## Slide 26 · 一句话总结

> GRPO = PPO 去 critic + group z-score baseline + rule reward = R1 算法核心。

---

## Slide 27 · 下一讲

**L03 R1-Zero** — 完整 R1-Zero 算法 + aha moment 涌现机制。

---

## Slide 28 · 检查清单

- [ ] 看懂 GRPO 公式 + 与 PPO 对比
- [ ] 推导 group advantage z-score
- [ ] 写下 KL 项 unbiased estimator
- [ ] 跑通 grpo_minimal.py（WSL2）
- [ ] minimal vs verl 一致性测试 PASS
