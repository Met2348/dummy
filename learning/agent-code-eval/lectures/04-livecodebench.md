# L04 · LiveCodeBench — 月度滚动防污染

**Jain et al. 2024** · arXiv 2403.07974

## 核心创新

每月**滚动 release** 新题，永远抓 train cutoff 之后的：
- LeetCode contests (weekly)
- Codeforces rounds
- AtCoder contests

→ **理论上不可能被训练污染**。

## 数据流

```
LeetCode 周赛 (每周日)
    ↓
LCB 团队抓题 + 测试
    ↓
按 release date 标注
    ↓
公开数据集（4 月题 / 5 月题 / 6 月题）
```

## 评测时怎么用

```python
# 报告 = (cutoff date) → 只用 cutoff 之后的题
results = run_lcb(model, after_date="2024-09-01")
```

GPT-4o (2024-10 cutoff) 在 11-12 月题分数 ≈ 6 月题（无污染优势）→ 体现真泛化。

## 分数（2024-2025）

| 模型 | LCB pass@1 (Hard) |
|------|-------------------|
| GPT-4o | 28% |
| Claude 3.5 Sonnet | 33% |
| Claude 3.7 Sonnet | 42% |
| o1-mini | 59% |
| **o1** | **68%** |
| **R1** | **65.9%** |

→ 推理模型（o1/R1）明显领先非推理。

## 4 类 task

| 类 | 来源 | 难度 |
|----|------|------|
| Generation | LeetCode Easy/Med | 主流 |
| Self-Repair | 给 broken code 改 | medium |
| Test Output | given code + input, predict output | hard |
| Code Execution | trace 执行 | medium |

## 实操

src/livecodebench_mock.py 3 题（two-sum / valid-parens / max-subarray）：

```python
from livecodebench_mock import run_livecodebench
from common import make_mock_model

rs = run_livecodebench(make_mock_model({}))  # empty → 0%
```

## 一句话

> LCB 月度滚动 = 永远新鲜的"周考"，测真泛化不是死记。
