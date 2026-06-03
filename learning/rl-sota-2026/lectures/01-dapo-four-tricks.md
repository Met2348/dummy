# L01 · DAPO 四件套 — ByteDance 2025.03 ⭐⭐⭐⭐⭐

> 32 slides | 80 min | RL SOTA 2026 第 1 讲

> ByteDance 2025.03 — Qwen-32B + DAPO 在 AIME 上达 50/100

---

## 学习目标

1. 看清 DAPO 在 GRPO 之上的 4 个独立 trick
2. 推导每个 trick 解决什么问题
3. 写出 4 件套独立 ablation 代码
4. 看每个 trick 的边际收益（消融实验）

---

## Slide 1 · DAPO 的成就

| 模型 | algo | AIME 2024 |
|------|------|----------|
| Qwen-32B base | - | 4/30 |
| Qwen-32B + GRPO | 标准 | 38/100 |
| **Qwen-32B + DAPO** | + 4 tricks | **50/100** ⭐ |

→ +12 pp 来自 4 件套，**每个 trick 贡献 2-3 pp**。

---

## Slide 2 · 4 件套总览

| # | Trick | 解决什么 |
|---|-------|--------|
| 1 | **Clip-Higher** | 探索性下降问题 |
| 2 | **Dynamic Sampling** | 全错/全对 group 浪费问题 |
| 3 | **Token-level PG Loss** | 长 response 训不动问题 |
| 4 | **Overlong Reward Shaping** | 长 response noise 问题 |

每个独立，可任意组合。

---

## Slide 3 · Trick 1 · Clip-Higher

GRPO/PPO 的 clip：`ratio ∈ [1-ε, 1+ε]`，对称。

观察：训练后期，模型熵下降，**正向 token 概率小**（比如 0.05），上 clip 阈值 1.2 → 实际上 limited 上升空间。

**修复**：
```
ratio ∈ [1-ε_low, 1+ε_high]
ε_low = 0.2, ε_high = 0.28
```

→ 不对称 clip，**允许正向 token 更激进地上升**。

---

## Slide 4 · Clip-Higher 数学

```python
ratio = torch.exp(log_probs_new - log_probs_old)
surr1 = ratio * A
surr2 = ratio.clamp(1 - eps_low, 1 + eps_high) * A
L_clip = -torch.min(surr1, surr2).mean()
```

差异：`clamp(0.8, 1.28)` 而非 `clamp(0.8, 1.2)`。

---

## Slide 5 · Clip-Higher 收益

ByteDance 实测：
- 训练前期：差异 < 1pp
- 训练后期（step > 4000）：+3-5pp accuracy
- entropy 高出 25%（更多探索）

→ **越训越后，Clip-Higher 越重要**。

---

## Slide 6 · Trick 2 · Dynamic Sampling

GRPO 每 prompt 固定 k 条 rollout。**问题**：
- 简单题 → k 条全对 → A = 0 → 无梯度
- 难题 → k 条全错 → A = 0 → 无梯度

**修复**：Dynamic Sampling
- 对每 prompt 持续采，直到至少 1 对 1 错
- 或者用 `pass@1` filter 后才进 batch

---

## Slide 7 · Dynamic Sampling 数学

```python
def collect_rollout(prompt, k_min=8, k_max=32):
    rollouts = []
    while True:
        new = rollout(prompt, n=k_min)
        rollouts.extend(new)
        rewards = [r for _, r in rollouts]
        if 0 < sum(rewards) < len(rewards):
            # 有对有错 → 可用
            break
        if len(rollouts) >= k_max:
            break
    return rollouts
```

→ 平均提高 rollout 利用率 30%。

---

## Slide 8 · Trick 3 · Token-level PG Loss

GRPO 默认 **response-level**：
```
L = mean over responses [ sum_t L_t / response_len ]
```

问题：长 response 每 token 权重 = `1/T`，**长 response 中的关键 token 几乎不更新**。

**修复**：token-level
```
L = mean over all (B*k*T) tokens [ L_t ]
```

→ 长 response 中的关键 token 同等权重，长 response 训得动。

---

## Slide 9 · Token-level vs Response-level 收益

| 设置 | response 长度 | accuracy |
|------|-------------|---------|
| response-level | 100 | 35% |
| response-level | 500 | 32%（长 response 学不动） |
| **token-level** | 500 | **42%** |

→ token-level 在长 response 上显著更好。

---

## Slide 10 · Trick 4 · Overlong Reward Shaping

某些 response 因为 `max_response_len` 截断 → 没生成 `</answer>`。

**朴素**：reward = 0（视为失败）→ 信号太严

**修复**：soft penalty
```
if response 完成 (有 </answer>):
    reward = base
elif response 因 截断 而未完成:
    reward = base · sigmoid((L_target - L_actual) / α)  # 部分分
```

→ 长 response 不被一刀切，保留学习信号。

---

## Slide 11 · Overlong Reward Shaping 公式

```python
def overlong_shaping(reward, response_len, target_len, alpha=200):
    if response_len < target_len:
        return reward
    # soft penalty
    return reward * torch.sigmoid(
        torch.tensor((target_len - response_len) / alpha)
    )
```

→ 在 target_len 之后 reward 逐渐衰减到 0，而不是突变。

---

## Slide 12 · 4 件套消融实验

ByteDance 公布数据（Qwen-32B + GRPO baseline）：

| config | AIME |
|--------|------|
| baseline | 38 |
| + Clip-Higher | 41 (+3) |
| + Dynamic Sampling | 43 (+5) |
| + Token-level PG | 47 (+9) |
| + Overlong Shaping | 50 (+12) |

→ **每个 trick 都有边际贡献**，4 件套累加。

---

## Slide 13 · 实现要点（与 GRPO 对照）

GRPO →  DAPO 的 4 处代码改动：

```python
# 1. Clip-Higher
surr2 = ratio.clamp(1 - eps_low, 1 + eps_high) * A    # eps_high > eps_low

# 2. Dynamic Sampling（在 rollout 阶段）
while not (有对 有错):
    继续采

# 3. Token-level PG (mask 后 mean over tokens, 不再除 response len)
L_clip = (L_clip_per_token * mask).sum() / mask.sum()

# 4. Overlong Shaping
rewards = [overlong_shape(r, L) for r, L in zip(rewards, response_lens)]
```

→ 全部加在 GRPO 框架上。

---

## Slide 14 · verl 配置（DAPO recipe）

```yaml
algorithm:
  adv_estimator: grpo
  clip_low: 0.2
  clip_high: 0.28          # ← Clip-Higher
  use_token_level_loss: true   # ← Token-level
data:
  rollout:
    dynamic_sampling: true   # ← Dynamic
    min_n: 8
    max_n: 32
reward:
  shaping:
    type: overlong_soft
    target_len: 4096
    alpha: 200
```

verl 0.4+ 内置 DAPO recipe，直接调。

---

## Slide 15 · 与 R1-Zero 的关系

R1-Zero = GRPO + rule reward + cold-start ✗。
DAPO = R1-Zero 路径上 + 4 件套优化。

→ 学好 GRPO 后，DAPO 是 4 个独立 trick，每个 30 分钟看懂。

---

## Slide 16 · 适用场景

| 任务 | DAPO 收益 |
|------|---------|
| 数学竞赛 (AIME) | +12pp ⭐ |
| 代码 (HumanEval+) | +5pp |
| GSM8K (相对简单) | +2pp |
| 开放对齐 | 几乎无 |

→ **DAPO 在难任务上收益最大**。

---

## Slide 17 · 与 VAPO / PRIME 的对比

| 算法 | 哪一年 | 核心 idea |
|------|------|---------|
| GRPO | 2024.02 | 去 critic + group |
| DAPO | 2025.03 | + 4 件套 |
| VAPO | 2025.04 | + Length-Adaptive GAE |
| PRIME | 2025.02 | + 隐式 PRM |

→ 全是 2025 H1 的 SOTA，DAPO 影响最大。

---

## Slide 18 · Capstone preview

L12 capstone：
- 基座：专题 5 capstone-A 训出的 R1-Zero baseline
- 4 件套 ablation：5 config（baseline / +1 / +2 / +3 / +4）
- 每 config 200 step，单 5090 24GB ~ 6h 总

→ 看每个 trick 单独贡献多少。

---

## Slide 19 · 工程坑

| 问题 | 修 |
|------|---|
| Clip-Higher 后 KL 爆 | β 调大 |
| Dynamic Sampling 慢 | max_n 设小一点 |
| Token-level loss 数值大 | adv normalize 在 batch 内 |
| Overlong Shaping 不平滑 | alpha 调小（更陡）|

---

## Slide 20 · 是否 4 件套都要

如果显存 / 时间紧：
- 必加：Token-level PG Loss（+9pp）
- 推荐：Clip-Higher（+3pp，零成本）
- 可选：Overlong Shaping（+3pp，看任务）
- 可选：Dynamic Sampling（+2pp，但实现复杂）

→ 优先 Token-level + Clip-Higher，简单且收益大。

---

## Slide 21 · 自测题

1. Clip-Higher 与对称 clip 的关系？
2. Dynamic Sampling 解决什么浪费？
3. Token-level vs Response-level PG loss 在什么长度差异最大？
4. Overlong Reward Shaping 为何用 sigmoid？
5. DAPO 4 件套各自的 wall-time 开销？

---

## Slide 22 · 历史与影响

DAPO (2025.03) 论文发布后：
- 4 件套很快被 verl / open-r1 / TinyZero 集成
- 每个独立 trick 被多篇论文引用
- Clip-Higher 成为新 baseline

→ 2025 H1 RL 工程标准升级。

---

## Slide 23 · 阅读建议

- **必读**：DAPO 论文 §4（4 件套细节）
- **必看**：verl DAPO recipe 源码
- **跳过**：附录的全部超参网格搜索（实操凭经验）

---

## Slide 24 · 实战入口

```bash
# 4 件套独立 ablation
python learning/rl-sota-2026/src/dapo_minimal.py

# verl 完整训练
python learning/rl-sota-2026/src/dapo_verl.py --config configs/dapo_full.yaml
```

---

## Slide 25 · L02 预告

**VAPO** = DAPO + Length-Adaptive GAE — 让 λ 随 response 长度动态调整。

---

## Slide 26-32 · 自测延伸题

（略，留作 reader exercise）

---

## 一句话总结

> DAPO = GRPO + 4 件独立 trick（Clip-Higher / Dynamic Sampling / Token-PG / Overlong Shaping）= 2025 RL SOTA 起点。
