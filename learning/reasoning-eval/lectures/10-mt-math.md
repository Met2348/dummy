# L10 · 工具增强数学推理

## 三种范式

### 1. Program-of-Thoughts (Chen 2022)

模型直接写 Python，外部执行。

```python
# Question: What is 17 * 23?
# Model output:
answer = 17 * 23
```

执行 `answer = 391`。

### 2. Tool-use (Toolformer, 2023)

模型在文本中插入 `<tool>...</tool>` 调用：

```
The total is <calculator>17 * 23</calculator>391 dollars.
```

外部 parser 解析 `<calculator>`，注入结果。

### 3. ReAct + math (Yao 2023)

```
Thought: I need to compute 17 * 23.
Action: calculator(17 * 23)
Observation: 391
Thought: So the answer is 391.
Final: 391
```

## 为什么对数学 bench 有效

LLM token 级别不擅长：
- 大数相乘（>3位易错）
- 复杂代数（化简 step 多）
- 高精度浮点

**外部 sandbox** 是 100% 准确的 → 把"算"外包。

## 分数对比 (GSM8K)

| 方法 | GSM8K |
|------|------|
| Vanilla CoT (GPT-3.5) | 57% |
| + PoT (code exec) | 77% |
| + ToRA (RL + tool) | 85% |
| GPT-4 + PoT | 95% |

→ 工具一般 **+15-20pp**。

## 关键安全：sandbox

LLM 输出代码 → 直接 `exec()` **危险**：
- `import os; os.remove(...)` 
- `open("/etc/passwd").read()`

防护：
1. **whitelist builtins**：只暴露 sum/range/abs/...
2. **AST parse 验证**：拒绝 import / open
3. **subprocess timeout**：5s 上限
4. **Docker/firejail/PyPy sandbox**：production-grade

## 实操

src/tool_aug_math.py：

```python
from tool_aug_math import run_with_tool
from common import make_mock_model

m = make_mock_model({"q1": "```python\nanswer = 17 * 23\n```"})
result = run_with_tool(m, "What is 17 * 23?", qid="q1")
# result = "391"
```

`_safe_exec_block` 阻断 `import`、`__*__`、`os.`、`exec(`、`eval(`、`compile(`、`open(`、`sys.`。

## 一句话

> 让模型写代码、sandbox 算 — 数学 bench 上的隐形 +15pp。
