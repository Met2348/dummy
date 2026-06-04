# L03 · BigCodeBench (2024)

**Zhuo et al. 2024** · arXiv 2406.15877

## 数据

- **1140 task**（vs HumanEval 164）
- **真实 API 调用**：每题用多个库（pandas, numpy, scipy, sklearn, BeautifulSoup, ...）
- **多步骤**：复杂业务逻辑
- 两种 split：
  - `Complete`: 函数补全
  - `Instruct`: NL → code

## 例题

```
Task: Read a CSV from URL, drop rows with missing values, plot a
histogram of a numeric column, return the column mean.

Required libs: pandas, requests, matplotlib
```

vs HumanEval 风格 "compute factorial" — **难度级别完全不同**。

## 设计原则

1. **真实工业 Python**：要用第三方库
2. **多 API 组合**：跨 库链式调用
3. **隐藏测试严格**：检查中间状态 + return value

## 分数

| 模型 | BCB Complete | BCB Instruct |
|------|--------------|--------------|
| GPT-3.5 | 36.0% | 26.4% |
| GPT-4 | 50.0% | 38.1% |
| GPT-4o | 51.1% | 39.4% |
| Claude 3.7 | 56.3% | 47.2% |
| DeepSeek-Coder V2 | 52.9% | 36.1% |

→ 仍远未饱和。

## 与 HumanEval 关系

```
HumanEval (饱和)
   ↓
BigCodeBench (难度+)
   ↓
LiveCodeBench (rolling 防污染)
   ↓
SWE-Bench (真实 issue 修复)
```

## sandbox 工程

BigCodeBench 需要预装 pandas/numpy/scipy/...
官方建议：Docker container per test。

## 教学版

我们没专门 runner（库依赖太重）。
推荐：跑 LiveCodeBench mock（L04）。

## 一句话

> BigCodeBench = "用 pandas 写真实业务" 的 LLM 考试。
