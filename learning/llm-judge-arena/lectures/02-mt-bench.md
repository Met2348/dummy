# L02 · MT-Bench (Zheng 2023)

**LMSYS** · arXiv 2306.05685

## 数据

- **80 prompt** × 8 类别（writing / roleplay / reasoning / math / coding / extraction / STEM / humanities）
- **2-turn 对话**：第 1 turn + 跟进 question
- 总 **160 turns** 评分

## 类别

```
writing   : 10 题 ("Write an email saying ...")
roleplay  : 10 题 ("You are a sci-fi writer ...")
reasoning : 10 题 ("If X then Y, then ...")
math      : 10 题 (GSM8K 风格)
coding    : 10 题 (LeetCode lite)
extraction: 10 题 ("From this passage, list ...")
STEM      : 10 题 (chem / bio / phy 知识)
humanities: 10 题 (history / law)
```

## 评分协议

GPT-4 当 judge：
```
Rate the response 1-10:
- 1-3: Major issues
- 4-6: Acceptable
- 7-8: Good
- 9-10: Excellent
Output: [[X]]
```

每 turn 一个分数 → 总分 = mean。

## 经典分数

| 模型 | MT-Bench |
|------|---------|
| GPT-3.5 turbo | 7.94 |
| GPT-4 | 8.99 |
| Claude 2 | 8.06 |
| LLaMA-2 70B-Chat | 6.86 |
| Vicuna-13B | 6.39 |
| **Claude 3.7 Sonnet** | **9.18** |
| **R1** | **9.0+** |

满分 10。

## 多 turn 价值

```
Turn 1: "Tell me about the French Revolution."
Turn 2: "Now expand on the Reign of Terror specifically."
```

测：
- 上下文记忆
- 连贯响应
- 主题扩展

单 turn 不能区分这类能力。

## 已知问题

1. **GPT-4 self-pref**：GPT-4 倾向给 GPT-4 高分
2. **80 题太少**：noise 大
3. **8 类别覆盖不均**：coding 只 10 题
4. **2024 起趋饱和**

## 升级版

- **MT-Bench 101** (2024)：101 题 + 多 turn 增强
- **WildBench** (2024.05)：1k+ 真实用户 prompt
- **MT-Bench Plus** (2024)：DeepMind 内部版

## 实操

src/mt_bench_runner.py：

```python
from mt_bench_runner import run_mt_bench, make_keyword_point_judge
from common import make_fixed_gen

gen = make_fixed_gen({"mt_1": "This is helpful step-by-step."})
judge = make_keyword_point_judge(["step", "helpful", "example"])
print(run_mt_bench(gen, judge)["overall"])
```

## 一句话

> MT-Bench = 2023-2024 对话模型"高考"，简单上手但 noise 大。
