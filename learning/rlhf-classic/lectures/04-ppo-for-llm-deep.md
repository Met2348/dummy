# L04 · PPO for LLM 深化

> 18 slides | 60 min | RLHF 三段管线第三段（核心）

---

## Slide 1 · LLM-PPO 与游戏-PPO 的不同

| 维度 | CartPole-PPO | LLM-PPO |
|------|-------------|---------|
| state | 4 维 vec | (B, T, V) tokens |
| action | 离散 | token id ∈ V (50k) |
| episode | 200 step | response 长度 (50-500) |
| reward | 每步 +1 | 终态 RM 分数 |
| critic | 独立 MLP | 与 actor 共享 backbone + value head |
| KL ref | 无 | **必须**（ref = SFT model） |

---

## Slide 2 · 4 model 协同

```
actor   π_θ          训 (用 LoRA 或 full)
critic V_φ           训 (与 actor 共享 backbone)
ref    π_SFT (freeze) KL 约束的"原点"
RM     r_ψ  (freeze)  rollout 后打分
```

显存：4 个 7B model = ~50GB → 显存敏感。

---

## Slide 3 · token-level reward 构造

```python
# 终态 RM 分数 + 每步 -β·KL
rewards[t] = -beta * (log π_act - log π_ref)[t]
if t == last_response_idx:
    rewards[t] += rm_score   # RM 终态 reward 加在最后一步
```

→ KL 平滑分布在每步，RM 只加在最后。

---

## Slide 4 · 为什么 KL 加在 reward 而非 loss

两种做法：
1. **Reward 减 KL** (InstructGPT): r' = r - β·KL → 通过 GAE 影响 advantage
2. **Loss 加 KL** (TRPO): L = L_PPO + β·KL → 直接梯度

实测 #1 更稳（GAE 平滑了 KL 信号）。

---

## Slide 5 · 自适应 β (adaptive KL ctrl)

```python
if KL > target * 1.5:
    beta *= 1.5
if KL < target / 1.5:
    beta /= 1.5
```

target = 6 是常见值。防止：
- β 太小 → policy 漂出 SFT 分布 → reward hacking
- β 太大 → policy 不动

---

## Slide 6 · GAE for LLM

对每条 response 独立算 GAE：
```python
for response in batch:
    rewards = build_token_rewards(...)  # (T_response,)
    values = critic_values[response_slice]
    adv = compute_gae(rewards, values, dones)  # GAE
```

注：γ 通常设 1.0（response 不太长，没必要 discount），λ = 0.95-0.99。

---

## Slide 7 · advantage normalization

每个 batch 内：
```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

**必须**：长 response 数值大，没 norm 会爆。

---

## Slide 8 · PPO clip loss (token-level)

```python
log_p_new = gather_logp(actor(input_ids), input_ids)
ratio = (log_p_new - log_p_old).exp()
surr1 = ratio * adv
surr2 = ratio.clamp(1-ε, 1+ε) * adv
pi_loss = -torch.min(surr1, surr2)
# mask response tokens
pi_loss = (pi_loss * response_mask).sum() / response_mask.sum()
```

ε = 0.2 是 default，DAPO 改为 (0.2, 0.28) 不对称。

---

## Slide 9 · value loss 也要 clip

```python
v_clipped = v_old + (v - v_old).clamp(-eps_v, eps_v)
v_loss = max((v - returns)^2, (v_clipped - returns)^2)
```

避免 critic 跳变。

---

## Slide 10 · entropy bonus

```python
entropy = -(p * log_p).sum(-1)
L_total = pi_loss + 0.5*v_loss - 0.01*entropy
```

entropy 系数过大 → policy 漫游；过小 → 早熟。LLM 通常 0.0-0.01。

---

## Slide 11 · rollout batch 设计

经验配置：
- mini_batch_size = 64 (token-level)
- rollout_size = 256 (response 数)
- ppo_epochs = 4
- num_mini_batches = 4
- total update / rollout = epochs × mini_batches = 16

→ 一次 rollout 喂 16 次 PG。

---

## Slide 12 · 显存与 grad accum

7B + 4 model + bf16：
```
actor + critic + grad + adam = ~30GB
ref + RM (frozen, no_grad) = ~14GB
KV cache during rollout = ~10GB
```

→ 80GB GPU 才宽松。小显存：critic LoRA + RM offload。

---

## Slide 13 · vLLM 加速 rollout

```python
# 慢：actor.generate() ~ 30s/256 responses
# 快：vllm engine = 1-2s/256 responses (15x)
```

vLLM 用 PagedAttention + tensor parallel。RLHF 框架 (OpenRLHF/verl) 默认集成。

---

## Slide 14 · 训练监控指标

| 指标 | 健康范围 |
|------|---------|
| reward (RM 分) | 上升 |
| KL (mean) | 0-10 |
| pi_loss | 围绕 0 波动 |
| v_loss | 下降 |
| entropy | 缓慢下降 |
| response_len | 适度变化 (无暴涨) |

异常：reward 涨但 KL 飞 → reward hacking。

---

## Slide 15 · 失败模式诊断

| 现象 | 诊断 |
|------|-----|
| reward 不动 | lr 太小 / β 太大 |
| reward 涨但生成乱 | β 太小，policy 漂 |
| KL 上窜 | β 跟不上，加 adaptive ctrl |
| critic loss 不降 | value head lr 单独调大 |
| OOM | 用 grad checkpoint / ZeRO-3 |

---

## Slide 16 · DPO 的替代视角

DPO 论文最大贡献：证明 RLHF 在 BT+KL 假设下等价于 closed-form loss，**不需要 RL**。
- 优势：1 个 model, 无 RM，无 PPO loop
- 劣势：只用 offline 数据，无 exploration
→ L01-L04 是基础，L05 后专题 3 DPO 家族会展开。

---

## Slide 17 · 三轨实现路径

```
ppo_llm_minimal.py    手写 4 model 协同 (本讲核心)
ppo_llm_trl.py        trl PPOTrainer (生产)
ppo_llm_verl.py       verl (DAPO/GRPO 时切，专题 5)
```

---

## Slide 18 · 一句话总结

> LLM-PPO = 4 model + token-level reward + KL ref + GAE + PPO clip。每个组件单独都简单，组合起来是工程艺术。

下一讲 L05 — RLHF 工程细节深挖。
