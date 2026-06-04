# L06 · Big-Bench Hard (BBH)

**Suzgun et al. 2022** · 23 task subset of Big-Bench

## 背景

Big-Bench 是 2022 年 Google 主导的"两百多任务大杂烩"。
BBH = 从中**挑出 23 个 large LM 都搞不定的硬骨头**。

## 23 任务全景

| 类别 | 代表任务 |
|------|---------|
| 推理 | `logical_deduction_three/five/seven_objects` |
| 时序 | `date_understanding`, `temporal_sequences` |
| 物体追踪 | `tracking_shuffled_objects_three/five/seven` |
| 因果 | `causal_judgement` |
| 几何/空间 | `geometric_shapes`, `navigate` |
| 词义 | `word_sorting`, `disambiguation_qa` |
| 形式 | `formal_fallacies`, `hyperbaton` |
| 算术 | `multistep_arithmetic_two` |

## 经典 demo: tracking_shuffled_objects

```
Alice has a red ball, Bob has a blue ball, Carol has a green ball.
Alice and Bob swap. Then Bob and Carol swap.
Who has the red ball?
A. Alice  B. Bob  C. Carol
```

要求模型**模拟状态演化**。

## CoT 对 BBH 提升巨大

```
Standard prompting    : 50.9%
CoT (chain-of-thought): 73.5%
```

→ **CoT 的发明就是为了这种任务**。

## 评测协议

- **3-shot CoT** 标准
- exact match（letter 或 verbatim 答案）
- 每任务 ~50-250 题

## Open LLM Leaderboard v2 用法

直接报 23 任务平均（去掉污染的 4 个 = 19 任务）。

## 实操：micro-BBH

src/bbh_runner.py 实现 3 任务：
- tracking (3 obj)
- date understanding
- logical deduction

```python
from bbh_runner import run_bbh, summarize
from common import make_random_model

rs = run_bbh(make_random_model())
print(summarize(rs))
# {'accuracy': 0.33, 'by_task': {'tracking': 0.5, 'date': 0.0, 'logic': 0.5}}
```

## 一句话

> BBH = 大模型的"奥数班"，没 CoT 几乎做不了。
