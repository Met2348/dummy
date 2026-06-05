# L08 · Judge 4 大 bias

## 1. Position bias

LLM judge 偏好 **position A**。

```
GPT-4 在 MT-Bench 上：
- A 位置胜率: 53%
- B 位置胜率: 47%
应该 50/50 (随机抽样下)
```

**对策**：
- 每对 prompt 两次（swap A/B），取一致 vote
- 或随机化提交顺序

## 2. Verbosity bias

LLM judge 倾向 **长答案 = 好**。

```
"the answer is 42." (10 chars)        vs
"The answer is 42, and here's why ..." (200 chars)
↑                                       ↑
short                                   long
                                        win 70%+
```

**对策**：
- AlpacaEval LC（length-controlled）
- Arena-Hard style-control

## 3. Self-preference

GPT-4 judge 给 **GPT-4 出的 response 高分**。

```
GPT-4 evaluating:
  GPT-4 vs Claude 3: 偏 GPT-4 +5%
  GPT-4 vs Llama-2: 偏 GPT-4 +15%
```

**对策**：
- 用第三方 judge（Claude → 评 GPT 和 Llama 谁好）
- 多 judge panel + 多数 vote

## 4. Style bias

LLM 偏好：
- markdown 格式（list / heading）
- "I'd be happy to help" 套话
- 长 introduction

```
Bad style:                  Good style:
"42."                       "I'd be happy to help! Here's the answer:\n\n**42**\n\nLet me know if you need more details."
```

**对策**：
- style-controlled regression
- 规则化拒绝套话

## 量化 bias 工具

src/judge_bias_demo.py：

```python
from judge_bias_demo import (
    position_bias_score, verbosity_bias_score, swap_consistency
)
from common import make_length_judge

j = make_length_judge(prefer_longer=True)
print(verbosity_bias_score(j))  # {'long_pick_rate': 1.0}
print(swap_consistency(j, n_pairs=20))  # 1.0 (length judge consistent)
```

## 实测值（GPT-4 judge on MT-Bench, 2024）

| Bias | 量化 |
|------|------|
| Position bias | 53% A-win (vs 50% no bias) |
| Verbosity | 70-80% 长答案胜 |
| Self-pref | +5-15pp |
| Style (markdown) | +3-8pp |

## 一句话

> LLM judge 4 大 bias = 位置 / 啰嗦 / 自卖自夸 / 套话风格。**没校正就别报数字。**
