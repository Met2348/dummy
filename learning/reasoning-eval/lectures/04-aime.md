# L04 · AIME 2024/2025 — 现代推理王者战场

## 什么是 AIME

American Invitational Mathematics Examination，美国数学奥赛二级（AMC → AIME → USAMO）：
- 每年 30 题（I/II 各 15 题），3 小时
- 答案是 **0-999 之间的整数**
- 高中生顶尖参赛

## 为什么成为 LLM 战场

1. **0-999 整数答案** → 评测简单（exact match）
2. **难度真的高**：人类参赛者平均 5-6/30
3. **未公开年份保护**：AIME 2024 / 2025 在训练 cutoff 后
4. **小数据**：30 题 → 单次 eval 便宜

## 关键分数 (2024-2025)

| 模型 | AIME 2024 |
|------|-----------|
| Claude 3.5 Sonnet | 16% |
| GPT-4o | 13% |
| **Claude 3.7 Sonnet** | **62%** (extended thinking) |
| **o1** | **83.3%** |
| **R1** | **79.8%** |
| **o3-mini high** | **87%** |
| AIME 满分 (30/30) | 100% |

→ R1 paper 主分数就是 AIME 2024。

## 评测细节（重要）

R1/o1 用 **maj@64**（64 次采样投票）而非 pass@1：

```
Score = max(maj@N for N in [1, 4, 16, 64])
```

为什么？
- AIME 答案空间小 (0-999)
- 多采样后投票稳定性高
- pass@1 太抖（一次错就 0）

## 实操

src/aime_runner.py：
```python
from aime_runner import run_aime, run_aime_passk
from common import make_dummy_model

# pass@1
rs = run_aime(make_dummy_model("0"))

# pass@k (4 samples)
pak = run_aime_passk(some_model, k=4)
# {1: ..., 4: ...}
```

## 评测陷阱

- "0-999 整数" 限制 → 模型可能输出 "1995"（超界）
- 处理：要么限制 generation，要么投票时只保留有效区间

## 一句话

> AIME 是当今"thinking model" 的最强 IQ 测试 — R1 时代的高考。
