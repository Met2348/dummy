# L09 · Test-Time Compute + RL

> 16 slides | 45 min | 推理时 scaling 新方向

---

## Slide 1 · 范式跃迁

```
2017-2023: train-time scaling — 模型变大
2024+:     test-time scaling — 推理变长
```

OpenAI o1 (2024.09) 首次明确：推理时给更多 token 也能涨能力。

---

## Slide 2 · 三种 test-time 方法

```
1. BoN (Best-of-N)        — 多 candidate 选最佳 (Topic 4)
2. CoT (Chain-of-Thought) — 推理变长
3. budget forcing         — 强制 think 更久 (s1)
```

---

## Slide 3 · s1 (Stanford 2025.01)

idea: 模型若早 stop (产 `</think>`)，强行注入 "Wait" 让它继续。
```
[</think> detected, total_think < budget]
    → ignore </think>, append "Wait, let me reconsider..."
    → 继续生成
```

→ 用极少代码 (~50 行) 实现 o1-style 长 think。

---

## Slide 4 · s1 数据

```
1k 精选数学题 + 长 CoT trace (来自 R1 distill)
SFT base model
推理时 budget forcing
→ MATH +12pp
```

→ "Less is more"：1k SFT + budget forcing 接近 R1。

---

## Slide 5 · Don't Overthink (2025)

警示论文：
- 推理 token 太多反而下降
- 模型陷入 "wait again, wait again..." 循环
- 真实 reasoning 与噪声混

→ budget 不是越大越好。

---

## Slide 6 · 最优 budget

实测 (DeepSeek-R1 数学):
| budget (think tokens) | accuracy |
|----------------------|---------|
| 0 (greedy) | 30% |
| 200 | 55% |
| 800 | 70% |
| 2000 | 75% |
| **8000** | **76%** ⭐ |
| 16000 | 73% (退化) |

→ 边际收益递减 + 过长退化。

---

## Slide 7 · Thinking-Optimal Scaling (2025)

观察：不同难度题最优 budget 不同。
- 简单题：50-100
- 中等：500-1000
- 难：2000-5000

→ 自适应 budget = 训练一个"难度估计器"。

---

## Slide 8 · 商业 thinking 模型

```
Claude 3.7/4 Extended Thinking: 默认 1024 budget,可配 max 64k
Gemini 2.5: thinking_budget 参数 (0 / 1k / 8k)
DeepSeek-R1: 推理 8k 默认
o1: 内部自动调
```

→ API 暴露 budget 参数已成共识。

---

## Slide 9 · 测试时 RL (TTRL)

idea: 不仅 train-time，**test-time 也做 RL**:
- 推理时 rollout N candidates
- 用 verifier 给 reward
- 单次 fine-tune (KL 强约束)
- 解新题

→ 极端 test-time scaling，但 latency 不可接受。

---

## Slide 10 · Test-time scaling laws

OpenAI o1 论文经验：
```
log(accuracy gain) ∝ log(test compute)
```

→ 线性投入算力得对数收益。但 train-time 也是对数收益，两条路并行。

---

## Slide 11 · 与 RL 训练的关系

```
Train-time RL: actor 学会"产长 CoT"
Test-time scaling: 强制 actor "用长 CoT"
```

两者互补。R1-Zero 是 train-time，s1 是 test-time。

---

## Slide 12 · 工程实现

```python
def generate_with_budget(model, prompt, min_think, max_think):
    output = ""
    in_think = True
    while in_think and tokens < max_think:
        chunk = model.generate_chunk(prompt + output, max_new_tokens=20)
        output += chunk
        if "</think>" in chunk:
            if tokens < min_think:
                output = output.replace("</think>", " Wait,")
                continue
            in_think = False
    return output
```

---

## Slide 13 · 适用任务

| 任务 | budget 收益 |
|------|-----------|
| 数学竞赛 | ⭐⭐⭐⭐⭐ |
| 代码 | ⭐⭐⭐⭐ |
| 长推理 | ⭐⭐⭐⭐ |
| 简单 chat | ⭐ (噪声) |
| 创作 | ⭐⭐ |

→ verifier-friendly 任务最赚。

---

## Slide 14 · 与 BoN 对比

```
BoN: 多 candidate 选最佳 → 并行高,token cost = N × len
TTC: 单 candidate 加长 → 串行,token cost = budget
```

经验：
- 简单题 BoN 更好
- 难题 TTC 更好
- 二者可组合 (BoN + budget)

---

## Slide 15 · 未来 (2026-2028)

预测：
- adaptive budget 标配
- model 自己估难度调 budget
- BoN + TTC 混合 search
- test-time RL (TTRL) 渐落地

---

## Slide 16 · 一句话总结

> Test-time scaling: 推理时给更多 token 也涨能力。s1 用 budget forcing 复刻 o1。商业模型已开放 budget 参数。

下一讲 L10 — 2026 商业 thinking models。
