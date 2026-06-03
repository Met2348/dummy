# L01 · ORM vs PRM

> 14 slides | 40 min | Process Reward 系列起点

---

## Slide 1 · 推理任务的 reward 难题

数学题：5 步推理，最后给答案。如何 reward？
- 看最后答案对错？→ ORM (Outcome RM)
- 每步打分？→ PRM (Process RM)

---

## Slide 2 · ORM (Outcome Reward Model)

```
input: (question, full_reasoning)
label: correct / incorrect (final answer)
loss : binary cross entropy
```

- 数据量小 (final 标注)
- 信号稀疏
- credit assignment 难 (5 步都对 vs 4 步对 1 步错)

---

## Slide 3 · PRM (Process Reward Model)

```
input: (question, partial_reasoning_up_to_step_k)
label: step_k 是 good / neutral / bad
loss : 3-way cross entropy
```

- 数据量大 (每步标注)
- 信号密集
- 失败溯源容易

---

## Slide 4 · Lightman 2023 "Let's Verify Step by Step"

OpenAI 第一个 large-scale PRM:
- PRM800K 数据：80 万步级 label
- 在 MATH 上 PRM > ORM (78% vs 72%)
- 验证 PRM 路线可行

---

## Slide 5 · PRM 训练

```python
class PRM(nn.Module):
    def __init__(self, lm):
        self.lm = lm
        self.head = nn.Linear(hidden, 3)   # good/neutral/bad

    def score(self, x, step_end_positions):
        h = self.lm(x).hidden_states[-1]
        return self.head(h[step_end_positions])
```

3-way classifier on step end hidden state。

---

## Slide 6 · 推理时聚合

每步 PRM 输出 prob_good：
- mean: 平均概率
- min: 任一步错 → 整体错
- **min_last**: 必须最后一步对 (Lightman 推荐)
- product: 链式

→ min_last 实测最好。

---

## Slide 7 · ORM vs PRM 对照

| 维度 | ORM | PRM |
|------|-----|-----|
| 数据成本 | 低 | 高 (8x) |
| 训练数据量 | 800 | 800k step |
| 信号 | 稀疏 | 密集 |
| 推理 BoN gain | +5pp | +15pp |
| 适用 | 简单 | 推理 |

---

## Slide 8 · 痛点：标注成本

PRM800K 每步专家标注：
- $20/h × 数万小时
- OpenAI 内部数据，未开源完整

→ 工业界需要自动方案 → Math-Shepherd。

---

## Slide 9 · Math-Shepherd 自动 PRM 数据

Wang 2024：
- 每个 step 做 N 次 MC rollout
- success rate > 0.7 → good
- success rate < 0.3 → bad
- 中间 → neutral

无需人工。

---

## Slide 10 · PPM (rStar-Math)

Microsoft rStar-Math: preference-based PRM
- 不打 good/bad，比较两条 partial reasoning 谁更好
- 用 BT loss 训
- 减少 label noise

---

## Slide 11 · PRIME — 隐式 PRM

THU+Microsoft 2025.02:
- 不训 PRM
- 用 actor / ref 差: r_t = β · (log π_actor - log π_ref)
- 自然得到 step 级信号

→ 训练 PRM 的"反义词"。

---

## Slide 12 · 应用：BoN reranking

```
1. 生成 32 candidates
2. 每条用 PRM 算 step scores
3. 用 min_last 聚合 → single score
4. 取最高分
```

实测 BoN(N=32) + PRM ≈ 10× greedy。

---

## Slide 13 · 应用：MCTS

PRM 当 value function 给 MCTS：
- expand: actor 提候选 step
- evaluate: PRM 给步分
- backprop: 更新 UCT

rStar-Math: Qwen-7B + MCTS + PPM = MATH 90%。

---

## Slide 14 · 一句话总结

> ORM 简单但信号差；PRM 强但贵；Math-Shepherd 自动化；PRIME 直接绕开。**核心是 step-level credit**。

下一讲 L02 — Let's Verify (Lightman 2023) 细节。
