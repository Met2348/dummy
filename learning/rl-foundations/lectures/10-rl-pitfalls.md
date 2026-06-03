# L10 · RL 陷阱合集

> 22 slides | 55 min | RL Foundations 系列第 10 讲

---

## 学习目标

合集 + 系统化前 9 讲遇到的所有"坑"：
- Reward Hacking 5 类
- 训练崩塌 4 因
- 调参敏感性 + RL 的"非可复现性"
- Reward Over-optimization（Gao 2022）

---

## Slide 1 · "RL 难"的 3 维

1. **数据非独立**：你的策略改变数据分布，不能像 SL 那样 train/val split
2. **reward 难定**：写不对 RM/规则就被 hack
3. **超参敏感**：lr × 2 可能从 500 reward 跌到 100

→ "RL 训练不像 SL，更像调炼丹炉"。

---

## Slide 2 · Reward Hacking 经典 5 类

| 类型 | 例 |
|------|---|
| 1. 重复 hack | "good good good good" (sentiment 95%) |
| 2. 复制 hack | response = prompt 原文复制 |
| 3. 关键词 hack | 反复出现 "amazing" 等正向 trigger |
| 4. 长度 hack | 句长 reward → 输出无意义 padding |
| 5. 结构 hack | format reward → 总是 `<think></think><answer></answer>` 但无内容 |

→ Goodhart's Law 实例化。

---

## Slide 3 · 防 Reward Hacking 4 策略

1. **KL ref penalty**：限制语言偏离 SFT 模型
2. **Constitutional check**：reward 不只看一项（多 RM 加权）
3. **adversarial RM**：再训一个 RM 专门检测 hacking
4. **人工 spot check**：定期看 sample

→ 工程上 1+4 最常用。

---

## Slide 4 · Reward Over-optimization（Gao 2022）

> **核心发现**：随着 RL 训练步数增加，gold reward 先升后降。

```
   gold reward (真实人类偏好)
        ╲
         ╲   ← over-optimization 区
          ╲
           ╲
            ╲
              proxy reward (RM)   ↑↑↑ 持续上涨
─────────────────────────────────────→ RL step
```

→ 用 RM 当 reward 训太久，会反向损害真实质量。

---

## Slide 5 · Over-optimization 修复

- **Early stop**：用真实人类偏好（金标）做验证 → reward 不升就停
- **Iterative RM**：每隔一段时间重新训 RM
- **KL penalty**：β 大一点限制偏离
- **Ensemble RM**：多 RM 加权平均

→ 这是专题 7 高级 RM 的核心话题。

---

## Slide 6 · 训练崩塌 4 因

| 症状 | 可能原因 |
|------|--------|
| reward 突然跌到底 | 梯度爆炸 / NaN / 模式塌缩 |
| reward 缓慢下降 | RM over-optimization |
| reward 卡死不动 | lr 太小 / advantage 全 0 / entropy 0 |
| 长度暴增 | 缺 length cap / length reward 比例失衡 |

---

## Slide 7 · Mode Collapse

症状：actor 输出多样性极低，几乎完全 deterministic。

原因：
- entropy 系数过小（< 0.001）
- 训练过早 deterministic
- 单一 reward signal 主导

修复：
- ent_coef = 0.01 起步
- 加 length penalty / format penalty 防过激
- 看 entropy 曲线，跌到 0 就停

---

## Slide 8 · 超参敏感性

PPO 的 5 个最敏感超参：
1. **lr**: 1e-5 vs 3e-5 差距巨大
2. **β (KL coef)**: 影响 actor 偏离 ref 程度
3. **ε (clip)**: 0.1 vs 0.3 训练稳定性差异
4. **K_epochs**: 4 vs 10 数据效率差异
5. **batch_size**: 4 vs 32 梯度噪声差异

→ 建议从 known-good config 起，每次只改一项。

---

## Slide 9 · 非可复现性

同一份代码 + 不同 seed → reward 差几十分常见。

原因：
- env stochasticity（CartPole 是 deterministic，但 LLM rollout 是 sample）
- network 初始化
- minibatch shuffle

应对：
- 3+ seed 跑 → mean ± std
- 看 std 而非单点
- 关键结论 5+ seed

---

## Slide 10 · KL 飞涨怎么救

监控 `KL(actor || ref)` per iter：

- 0.01 - 0.1：健康
- 0.1 - 0.5：警戒，β 加大
- > 0.5：危险，可能崩

应对：
1. β ×= 1.5
2. lr ÷= 2
3. K_epochs ÷= 2
4. 看样本，是否已 hack

---

## Slide 11 · 长度暴增怎么救

LLM-PPO 常见现象：mean response length 翻倍。

原因：长 response 累积更多 token-level reward。

修复：
1. **Length cap**：`max_new_tokens = 50` 硬限制
2. **Length penalty**：reward -= 0.01 × len（小系数即可）
3. **Length-normalized loss**：把 PPO loss 按 response len 归一化（SimPO 思路）
4. **DAPO Overlong Shaping**：长 response soft penalty 函数

---

## Slide 12 · Critic 学不动怎么办

`L_vf` 持续高 ≠ 收敛 → critic 不行 → advantage 错 → actor 也错。

排查：
1. critic lr 是否同 actor 一致？（一般独立）
2. value head 初始化是否合理？（gain=1.0 standard）
3. observation 范围是否 reasonable？
4. K_epoch 不要太大，critic 也容易过拟合

→ critic 单独训前 100 iter 可加速收敛。

---

## Slide 13 · 数据高效用：multi-K epoch

PPO 标准 K=4~10。但 K 太大会有问题：
- old policy 与 new 差距大 → IS 估计不准
- clip 频繁触发 → "无效"更新

→ 经验：CartPole K=10 OK；LLM-RL K=2~4 起步。

---

## Slide 14 · 数据分布漂移

on-policy RL 的核心痛点：训练数据来自当前 policy → 改 policy → 数据失效。

对策（按彻底程度）：
1. PPO clip：限制单步漂移
2. KL penalty：限制累积漂移
3. Replay buffer + IS：off-policy 修正（DQN 风）

LLM-RL 主流是 1+2，因为简单且够用。

---

## Slide 15 · 多目标 RL 的陷阱

实际 RLHF 需要 helpful + harmless 等多 RM：

```
reward = 0.7 · R_help + 0.3 · R_harm
```

但 R_help 和 R_harm 单位/范围不同 → 加权失衡。

解法：
- 归一化每个 R 到 [0, 1]
- 多目标 Lagrangian（Safe-RLHF 风格）
- Pareto 多重 PPO

→ 专题 2 多目标 RLHF 详讲。

---

## Slide 16 · Spurious Rewards（2025 警示）

(剧透：专题 5 L12 详讲)

研究者用**随机 binary reward** 训 Qwen + GRPO，居然 accuracy 涨 21pt。

原因（猜想）：
- Qwen 本身已 partly trained on math
- 任何 RL signal 都激活了 latent 数学能力
- 改变了 self-explore 行为

→ **教训：reward 涨不等于真涨**。要用真实人类偏好 / 金标验证。

---

## Slide 17 · RL Pitfalls 清单（汇总）

| # | 陷阱 | 解决工具 |
|---|------|--------|
| 1 | reward hacking | KL penalty / spot check |
| 2 | mode collapse | entropy bonus |
| 3 | KL 暴涨 | adaptive β |
| 4 | length 暴增 | length penalty |
| 5 | critic 学不动 | c_v 调整 + 预训 critic |
| 6 | 训不动 | lr / adv norm |
| 7 | RM over-optimization | early stop |
| 8 | reward 不准 | RM ensemble / iterative |

---

## Slide 18 · LLM-RL 专用陷阱

1. **tokenization 不一致**：actor 与 RM 用不同 tokenizer 时
2. **EOS token 漏处理**：导致 response 实际长度算错
3. **left-padding 与 right-padding 混用**：generate 用 left, train 用 right
4. **gradient_checkpointing 与 generate 不兼容**：要切换
5. **bf16 vs fp16 的 NaN 风险**：bf16 更稳

---

## Slide 19 · 一份 known-good 配置（GPT-2-small + sentiment）

```python
config = {
    "model": "gpt2",
    "lr": 1e-5,
    "batch_size": 8,
    "ppo_epochs": 4,
    "cliprange": 0.2,
    "vf_coef": 0.1,
    "init_kl_coef": 0.05,
    "target_kl": 6,
    "adap_kl_ctrl": True,
    "max_new_tokens": 20,
    "grad_clip": 1.0,
}
# 5090 24GB 上 100 iter ≈ 30 min，sentiment 从 0.45 → 0.7
```

---

## Slide 20 · 自测题

1. Reward Over-optimization 与 Reward Hacking 区别？
2. 训练崩 4 因，按出现概率排序？
3. PPO 5 个最敏感超参？
4. mode collapse 与 KL 暴涨可以同时发生吗？为何？
5. Spurious Rewards 给我们什么教训？

---

## Slide 21 · 调试清单（口袋版）

PPO 不收敛？按顺序：

- [ ] adv 是否归一化？
- [ ] lr 是不是太大？ (3e-5 → 1e-5)
- [ ] β 是不是太小？ (0.01 → 0.05)
- [ ] entropy 是不是塌缩？看 ent 曲线
- [ ] KL 是不是飞涨？看 KL 曲线
- [ ] length 是不是飞涨？看 mean_len 曲线
- [ ] critic L_vf 是不是降？看 vf 曲线
- [ ] 看 5 个 sample，是不是 reward hacking？

---

## Slide 22 · 下一讲

**L11 capstone** — IMDb 完整 sentiment PPO + 三段管线雏形。
