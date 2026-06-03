# L12 · Capstone — GSM8K PRM + BoN

> 12 slides | 30 min | Process Reward 工具箱收官

---

## Slide 1 · Capstone 目标

完整 PRM + BoN 推理 pipeline 在 GSM8K：
- 训 PRM (Math-Shepherd 自动数据)
- Qwen2.5-0.5B 每问 32 candidates
- PRM rerank → 评估

预期：PRM rerank ≥ greedy +10pp。

---

## Slide 2 · 数据准备

1. **base data**: GSM8K train (~7.5k)
2. **MC rollout**: 每 step 8 次后续 rollout
3. **label**: success_rate > 0.7 → good
4. **size**: 约 5k 步级 label (~1k 题 × 5 步)

```bash
python math_shepherd_data_gen.py --questions gsm8k_train.jsonl --out prm_labels.jsonl
```

---

## Slide 3 · PRM 训练

```yaml
base: Qwen2.5-0.5B
head: Linear(896, 3)   # good/neutral/bad
data: 5k step labels
loss: 3-way cross entropy
lr  : 1e-5
ep  : 3
time: ~1h on 5090
```

---

## Slide 4 · PRM 评估 (held-out)

- step accuracy > 70% (3-way)
- step F1 (good vs not-good) > 0.65

不达标 → 检查 label noise / 增数据。

---

## Slide 5 · Stage 2: 推理 pipeline

```python
def solve(question, actor, prm, k=32):
    candidates = actor.generate(question, num_return_sequences=k,
                                 do_sample=True, temperature=0.7)
    scores = []
    for c in candidates:
        step_logits = prm.score_steps(c, parse_step_ends(c))
        scores.append(aggregate_step_scores(step_logits, "min_last"))
    return candidates[argmax(scores)]
```

---

## Slide 6 · 对比基线

```
greedy:        actor.generate(temperature=0)
majority:      32 candidates 投票
BoN-PRM:       32 candidates + PRM rerank
BoN-weighted:  32 candidates + score 加权投票
```

---

## Slide 7 · 评估指标

GSM8K test 100 题：
| 策略 | 预期 acc |
|------|---------|
| greedy | 25% (Qwen2.5-0.5B base) |
| majority(32) | 35% |
| BoN-PRM(32) | **40%** ⭐ |
| BoN-weighted | 38% |

---

## Slide 8 · 关键超参

```yaml
n_candidates: 32       # 多 1 倍线性
temperature: 0.7       # 太低多样性差
top_p: 0.95
prm_aggregation: min_last  # 不要 mean
```

---

## Slide 9 · 常见失败

| 现象 | 修 |
|------|---|
| PRM acc < 60% | data 质量检查 + 增数据 |
| BoN 提升 < 5pp | n_candidates ↑ / temp ↑ |
| 所有 candidate 同答案 | temperature ↑ / top_p ↑ |
| PRM 偏向长 response | aggregate 用 min_last |

---

## Slide 10 · 进阶：MCTS + PRM

> rStar-Math 路径：MCTS 扩展 + PRM 当 value → 90% MATH。

简化版：
```python
mcts = MCTSNode(question)
for _ in range(100):
    leaf = mcts.expand(actor)   # actor 提候选
    score = prm.score(leaf)     # PRM 评分
    mcts.backprop(score)
best = mcts.best_path()
```

---

## Slide 11 · 与 R1-Zero 对比

| 维度 | PRM + BoN | R1-Zero |
|------|-----------|---------|
| 训练 | PRM | actor |
| 推理 | N candidates | 1 candidate (long CoT) |
| reward | PRM | rule (verifier) |
| 提升来源 | search | learn |

→ PRM + BoN 是 search-time，R1 是 train-time。

---

## Slide 12 · 一句话总结

> PRM + BoN = 推理时 scaling 经典做法。+10pp 仅需 1h 训练 + 32× 推理算力。

🎓 **Topic 4 Process Reward 完结。**
下一专题 5 — R1 时代（系列高峰）。
