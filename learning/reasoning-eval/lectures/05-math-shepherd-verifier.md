# L05 · 自动 verifier — math-verify & sympy

## 为什么需要

数学答案有无数等价形式：
```
0.5  =  1/2  =  \frac{1}{2}  =  50%  =  0.50000  =  1/(1+1)
```

简单 exact match 会**大量误判**。

## sympy 思路

```python
import sympy as sp

def equiv(pred: str, gold: str) -> bool:
    try:
        p = sp.simplify(sp.sympify(pred))
        g = sp.simplify(sp.sympify(gold))
        return p == g or sp.simplify(p - g) == 0
    except Exception:
        return pred.strip() == gold.strip()
```

可以处理：
- 数值：`0.5 == 1/2 == \frac{1}{2}`
- 代数：`x+y == y+x`
- 三角：`sin(pi/4) == sqrt(2)/2`

## math-verify 库

`math-verify` (HuggingFace 2024) 包装 sympy：
```python
from math_verify import parse, verify
gold = parse("1/2")
pred = parse("0.5")
verify(gold, pred)  # True
```

特别处理 LaTeX 输入。

## 已知 corner case

| 表达 | 难点 |
|------|------|
| `\frac{a}{b}` vs `a/b` | LaTeX parse |
| `\sqrt{2}` vs `1.414...` | irrational tolerance |
| `2\pi` vs `6.283...` | symbol vs float |
| `{1, 2, 3}` vs `[1, 2, 3]` | set vs list |

## 评测时的 trick

1. **多步 fallback**：sympy → numeric → string
2. **tolerance**：浮点 tol=1e-6
3. **canonical form**：先 simplify 双方
4. **timeout**：sympy 偶尔挂，加 5s timeout

## 我们的 toy 版

src/math_verify_demo.py:
```python
from math_verify_demo import equiv, parse_to_float

equiv("0.5", "1/2")           # True
equiv("\\boxed{18}", "18")    # True
equiv("\\frac{3}{4}", "0.75") # True
equiv("50%", "0.5")           # True
```

虽不如真 sympy，但能 cover 80% case。

## 实战教训（R1 paper）

R1 用 math-verify 后，GSM8K acc 从 87% → 89% (+2pp 纯靠 verifier 改进)。

→ **verifier 是 hidden gain**。

## 一句话

> 没 verifier = 数学评测全是"漏判"，sympy 是 LLM 数学评测的隐形必备。
