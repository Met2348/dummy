# L09 · 逻辑推理 — ZebraLogic / BBH-logic

## ZebraLogic (2024)

经典 "Einstein riddle"（zebra puzzle）扩展：

```
There are 5 houses in a row. Each painted different color,
each person of different nationality, each drinks different beverage,
each smokes different brand, each owns different pet.
The Brit lives in the red house. The Swede has a dog. ...
Who owns the zebra?
```

题数：1000+ 自动生成。

## 难度阶梯

- **2x2 grid**: 2 人 × 2 属性 → 大模型 ~95%
- **3x3 grid**: 3×3 → ~70%
- **4x4 grid**: 4×4 → ~50%
- **5x5 grid (经典 Einstein)**: → top 30%, R1 ~50%

随 grid size 指数增长难度。

## 评测

- 答案 = 单个属性值（"zebra owner"）或全 grid
- exact match

## 为什么是 LLM "硬骨头"

- **组合空间大**：5×5 grid → 120^5 ≈ 2.5×10^10 状态
- **必须搜索/SAT**：纯文本推理几乎不可能
- **CoT 不够**：需要回溯（backtracking）

## BBH 逻辑子集

L06 提过的 BBH 23 任务中 5 个是逻辑：
- `logical_deduction_three/five/seven_objects`
- `causal_judgement`
- `formal_fallacies`

它们都是 ZebraLogic 的小型变体。

## 实操

src/zebra_logic.py 3 题（2×2 + 3×3）：

```python
from zebra_logic import run_zebra
from common import make_dummy_model

rs = run_zebra(make_dummy_model("Alice"))
```

## 一句话

> 逻辑 bench = LLM 的"数独考试"，CoT 帮你做小题，做大题要工具。
