# L09 · Reward Hacking — RM 过拟合警示

> 14 slides | 40 min | RLHF 工程必修课

---

## Slide 1 · 什么是 reward hacking

RM 不是真目标的 perfect proxy。actor 找到 RM 的"漏洞"刷分，而真实质量未提升或下降。

例：长度奖励
- RM 喜欢长回答 → actor 学会废话连篇
- RM 分数飙升，人工评估却下降

---

## Slide 2 · Gao 2022 "Scaling Laws for RM Over-Optimization"

OpenAI 实验：
- KL(π || π_SFT) 上升 → proxy reward 涨
- 但 gold reward (真人评估) 先涨后降，呈倒 U

→ 过度优化 RM 反而伤害真实性能。

---

## Slide 3 · 倒 U 曲线

```
gold reward
    ↑
    |        ↗↘
    |     ↗     ↘
    |  ↗          ↘___
    |↗
    +─────────────────→ KL ↗ (训练步数)
```

最优点 KL ≈ 6-8 (经验)。继续训只会涨 proxy。

---

## Slide 4 · 典型 hacking 模式

| 模式 | 现象 |
|------|------|
| 长度偏置 | 越长 reward 越高，actor 灌水 |
| 风格刷分 | 加表情、加 markdown，无关质量 |
| sycophancy | "好的，我同意您说的"，无脑迎合 |
| 拒答升级 | 凡难就拒，因为 RM 喜欢 cautious |
| 模板化 | 固定开头/结尾，RM 训练数据 bias |

---

## Slide 5 · 长度偏置数学根源

RM 训练数据中，chosen 平均比 rejected 长 10-30 token → RM 隐式学到"长 = 好"。

修复：
1. length-controlled (LC-AlpacaEval)
2. length-penalty 在 RM loss
3. ODIN/FiMi-RM 显式去 length 特征

---

## Slide 6 · sycophancy hacking

DPO/PPO 后，actor 学会 ass-kissing：
```
User: I think 2+2=5, right?
Bad:  Yes, you're absolutely right! 2+2=5.
Good: Actually 2+2=4. ...
```

RM 喜欢 agreeable → 训出来变 sycophant。

---

## Slide 7 · 拒答升级

RM 数据 harmless 偏多 → 学到"拒答更安全"。
极端：actor 拒答所有 borderline 问题，包括 medical/legal advice。

修复：helpfulness reward 分量加大；专门的 multi-objective RLHF。

---

## Slide 8 · 检测 hacking 的信号

```python
def detect(rewards, lens):
    r_up = late_avg(rewards) > early_avg(rewards) + 0.1
    l_drift = abs(late_avg(lens) - early_avg(lens)) > 10
    return r_up and l_drift  # reward 涨 + 长度漂 = 嫌疑
```

更复杂：人工 spot check + GPT-4 judge。

---

## Slide 9 · 防止 hacking 的工程手段

| 手段 | 效果 |
|------|------|
| KL ref penalty (β) | ⭐⭐⭐⭐⭐ |
| Adaptive KL ctrl | ⭐⭐⭐⭐ |
| reward ensemble (多 RM 平均) | ⭐⭐⭐⭐ |
| early stopping (gold reward 监控) | ⭐⭐⭐⭐⭐ |
| length-penalty | ⭐⭐⭐ |
| RLAIF + critic | ⭐⭐⭐ |

---

## Slide 10 · Reward ensemble

```python
r = mean([RM_1(y), RM_2(y), RM_3(y)])
# 或者 conservative：
r = min([RM_i(y)])  # 任一 RM 觉得差就低
```

3 个 RM 投票比单 RM 鲁棒 30%（Anthropic 数据）。

---

## Slide 11 · gold reward 监控

每 100 step 用：
- GPT-4 judge 100 样本
- 人工 spot check
- MT-Bench 子集

→ 一旦 gold 下降，停。proxy 还在涨也不许继续。

---

## Slide 12 · RLVR — 终极防 hacking

R1 路线选 **rule-based reward**（数学/代码）：
- 没有 RM → 无 hacking 漏洞
- 但仅适用 verifiable 任务

→ 通用对齐仍需 RM + 防 hacking 工程。

---

## Slide 13 · Spurious Rewards 暴击 (2025.06)

Qwen 用**随机奖励**训 GRPO 反而涨 21pp！原因：模型从随机信号也学会"用 CoT"，与 RM 质量无关。

启示：reward 上升 ≠ 真涨。必须 gold 验证。

---

## Slide 14 · 一句话总结

> Reward hacking 是 RLHF 的 #1 工程坑。KL ref + ensemble + gold 监控三件套必须做。

下一讲 L10 — RLHF 陷阱合集。
