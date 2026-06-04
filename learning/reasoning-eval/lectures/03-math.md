# L03 · MATH — Hendrycks 竞赛数学

**Hendrycks et al. 2021** · arXiv 2103.03874

## 数据

- **12500 problems** (7500 train + 5000 test)
- **5 难度**：Level 1 (最易) → Level 5 (AMC/AIME)
- **7 科**：Algebra / Counting & Prob / Geometry / Intermed Algebra / NumberTheory / Prealgebra / Precalculus
- 答案格式：`\boxed{...}` LaTeX

## 例题（Level 4）

```
Q: Find the smallest positive integer n such that 3^n ≡ 1 (mod 80).
Solution: We have 3^1=3, 3^2=9, 3^3=27, 3^4=81≡1 (mod 80).
So n = \boxed{4}.
```

## 评测协议

1. **answer extract**: `\boxed{...}` regex
2. **equivalence check**:
   - 简单：字符串相等
   - 进阶：sympy parse 后 equiv（`1/2 == 0.5 == \frac{1}{2}`）
3. **`math-verify`** lib：业界事实标准

## 经典分数

| 模型 | MATH (pass@1) |
|------|--------------|
| GPT-3 175B (CoT) | 5.2% |
| Minerva 540B | 50.3% |
| GPT-4 | 50.6% |
| Claude 3.7 | 71% |
| R1-Zero | 71.5% |
| R1 | 79.8% |
| o1 | 94.8% |

注：**MATH Lvl 5** 是 Open LLM Leaderboard v2 用的子集（最难）。

## 与 GSM8K 区别

| 维度 | GSM8K | MATH |
|------|-------|------|
| 难度 | 小学 | 竞赛 |
| 答案 | 整数 | 数 / 分数 / 表达式 |
| 解答 | 自然语言 | LaTeX |
| 评判 | exact match | sympy equiv |

## 实操

src/math_runner.py 6 题 micro：
```python
from math_runner import run_math, by_level
rs = run_math(some_model)
print(by_level(rs))  # {1: ..., 2: ..., 3: ..., 4: ..., 5: ...}
```

src/math_verify_demo.py 处理 `\boxed{}` / `\frac{}{}` / `50%`：
```python
from math_verify_demo import equiv
equiv("0.5", "\\frac{1}{2}")  # True
equiv("\\boxed{18}", "18")    # True
```

## 一句话

> MATH 是 GSM8K 的"高中升级版"，sympy 验证是关键技术细节。
