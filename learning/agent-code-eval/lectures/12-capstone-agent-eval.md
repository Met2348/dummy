# L12 · Capstone — mini-agent 5-bench 联跑

## 目标

把 Topic 3 内容串起：HumanEval + LiveCodeBench + SWE-Bench + WebArena + BFCL，
用 2 个 mock model 对照。

## 设计

```
src/mini_agent.py
├── run_all(model)  → 5 bench score dict
└── to_md(scores, label) → markdown table
```

## 跑

```python
from mini_agent import run_all, to_md
from common import make_mock_model

# empty model
empty = make_mock_model({}, default="")
print(to_md(run_all(empty), "empty_baseline"))

# smart model (gold answers)
# ... 见 mini_agent._self_test
```

预期：
```
# Mini-agent capstone — `smart_model`
| bench | score |
|---|---:|
| humaneval     | 1.00 |
| livecodebench | 1.00 |
| swebench      | 1.00 |
| webarena      | 1.00 |
| bfcl          | 1.00 |
| **avg** | **1.00** |
```

## 想表达什么

1. **5 类 bench 联跑**：覆盖 code/SWE/Web/Tool
2. **统一接口**：所有 runner 接 `ModelFn = Callable[[str, int], str]`
3. **可替换为真模型**：只需改 `make_mock_model` → 真 inference 函数

## 真模型替换示例

```python
import openai

def gpt4o(prompt: str, max_new_tokens: int) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role":"user","content":prompt}],
        max_tokens=max_new_tokens,
    )
    return resp.choices[0].message.content

scores = run_all(gpt4o)
print(to_md(scores, "gpt-4o"))
```

## 退出条件

- 5 个 bench 都返回 0-1 之间
- empty model avg = 0.0
- oracle model avg = 1.0
- self_test PASS

## 一句话

> 5 bench 联跑 = LLM agent 能力的"五线综合考评"。
