# L02 · HumanEval + MBPP

## HumanEval (Chen 2021, OpenAI)

- **164 Python 函数补全题**
- 每题：signature + docstring + hidden tests
- 评测：pass@k（k=1, 10, 100）

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Return True if any two numbers are closer than threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """
    # 模型补全
```

## MBPP (Austin 2021, Google)

- **974 简单 Python 题**
- 每题：NL 描述 + 3 test cases
- 比 HumanEval 题更容易、覆盖更广

```
Task: Write a function to find the maximum of three numbers.
Tests:
  assert max3(1,2,3) == 3
  assert max3(5,1,2) == 5
```

## 经典分数

| 模型 | HumanEval pass@1 | MBPP pass@1 |
|------|------------------|-------------|
| GPT-3.5 | 48% | 52% |
| GPT-4 | 67% | 67% |
| Claude 3.7 | 89% | 90% |
| Codex (12B) | 28.8% | — |
| StarCoder 15B | 33% | 47% |
| DeepSeek-Coder V2 | 90.2% | 89% |

→ 几乎全饱和（top >90%）。

## Pass@k 公式

```
pass@k = E_problems [ 1 - C(n-c, k) / C(n, k) ]
```

其中 n=样本数、c=正确样本数。
推荐 n ≥ 100, k=1。

## sandbox 必备

`exec(generated_code)` 危险：
- `import os; os.remove(...)`
- 无限循环耗 CPU
- 文件 I/O

防护栈：
1. **限制 builtins**：只暴露安全函数
2. **AST parse 验证**：阻断 import / open
3. **subprocess + timeout**：python -c with 2-5s
4. **Docker / firejail**：production

## 我们的 toy sandbox

src/common.py `safe_exec()`：
```python
FORBIDDEN = ("import os", "import sys", "__import__", "open(", ...)
```

只暴露白名单 builtins，没文件访问。

## 实操

```python
from humaneval_runner import run_humaneval, run_passk
from common import make_mock_model

# 用 reference 答案构造 oracle
refs = {t["qid"]: f"```python\n{t['ref']}\n```" for t in _TASKS}
oracle = make_mock_model(refs)
print(run_passk(oracle, k=4))
# {'pass@1': 1.0, 'pass@4': 1.0}
```

## 一句话

> HumanEval/MBPP 是 LLM 代码能力的"小学考试"，今天全饱和但教学价值仍在。
