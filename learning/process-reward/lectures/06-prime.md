# L06 · PRIME — 隐式 PRM 从 outcome 自学步级

> 14 slides | 40 min | Process Reward 神来之笔

---

## Slide 1 · 动机

PRM 训练贵：
- PRM800K 需 80 万人工 step label
- Math-Shepherd 自动但需大量 rollout

**PRIME 问**：能不能不训 PRM？

---

## Slide 2 · 核心 idea

**actor 与 ref 的差**就是 step 级信号。
```
r_t = β · (log π_actor(y_t) - log π_ref(y_t))
```

- actor 在好步上概率拉高 → r_t > 0
- actor 在坏步上下降 → r_t < 0

→ 自然出现 step 级 reward。

---

## Slide 3 · 数学直觉

DPO 公式：implicit reward = β·log π/π_ref。
PRIME = "把这个 implicit reward 用在每个 token / step 上"。

ref = SFT model，是"未优化"基线；actor 是"优化中"，差异即"学到了什么"。

---

## Slide 4 · 训练流程

```
1. 初始 actor = SFT (≡ ref)
2. rollout N 条响应
3. outcome reward (正确/错) 算 advantage
4. per-token reward = β · (log π_actor - log π_ref)
5. 聚合到 step → step advantage
6. PPO update
```

不需要单独 PRM model。

---

## Slide 5 · 与 GRPO 关系

GRPO advantage = response-level z-score。
PRIME 在 GRPO 之上加 step 级信号：
- A_response_level (GRPO 标配)
- + A_implicit_step (PRIME 增量)

最终：A_t = α·A_resp + (1-α)·A_step。

---

## Slide 6 · 为何"自学步级"成立

直觉：
- actor 在"正确推理路径"上拉高 → ref 没拉 → log ratio +
- actor 在"错路径"也拉但 outcome 错 → ref 也降 → log ratio 小

→ 隐式 reward 与 step 质量自然相关。

---

## Slide 7 · 关键超参 β

```
β 太大: implicit reward 主导，actor 易飞
β 太小: 等于 GRPO baseline
β 推荐: 0.01 ~ 0.05
```

PRIME 论文用 0.04。

---

## Slide 8 · 与 KL penalty 反号

注意:
```
KL penalty = -β · KL(π||π_ref) = -β · (log π_actor - log π_ref)
PRIME implicit reward = +β · (log π_actor - log π_ref)
```

正好反号！PRIME 把 KL 信号变成 reward（exploitation 倾向）。

→ β 不能太大，否则失去约束。

---

## Slide 9 · 实测效果

PRIME 论文 Eurus-2-7B + MATH:
| 方法 | accuracy |
|------|---------|
| SFT | 30 |
| GRPO | 38 |
| GRPO + 训 PRM | 43 |
| **PRIME** | **45** ⭐ |

→ 不训 PRM 反而比训 PRM 好（避免 PRM noise）。

---

## Slide 10 · 工程优势

| 维度 | 训 PRM | PRIME |
|------|--------|-------|
| 额外 model | 1 | 0 |
| 数据准备 | 5k-50k step | 0 |
| 显存 | +7GB (7B PRM) | 0 |
| 实现 | 100 行 | 20 行 |

---

## Slide 11 · 实现

```python
def implicit_prm_per_token(log_p_actor, log_p_ref, beta=0.05):
    return beta * (log_p_actor - log_p_ref)

def aggregate_to_step(per_token_r, step_end_positions):
    cumsum = per_token_r.cumsum(-1)
    return [cumsum[pos] - cumsum[prev] for prev, pos in pairs]
```

← 这是 PRIME 全部核心。

---

## Slide 12 · 局限

PRIME 要求 actor 与 ref **同 backbone**：
- 用了 ref 副本（与 DPO 一样）
- 显存仍 ×2

但比训 PRM 仍便宜很多。

---

## Slide 13 · 与 R1 路线的关系

R1-Zero: rule-based reward (verifier)，无 PRM，无 PRIME。
PRIME: 仍用 outcome reward 但加自学 step 信号。

→ PRIME 是 R1-Zero 与 PRM 的中间方案。

---

## Slide 14 · 一句话总结

> PRIME = "不训 PRM 也能得到 step 级 reward"。actor/ref log ratio = 隐式 step value。

下一讲 L07 — RLVR 程序化 reward。
