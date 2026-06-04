# L09 · BFCL — Berkeley Function Calling

## 数据

- **2k+ 测试**横跨：
  - Simple: 1 函数选 1 调
  - Multiple: N 函数选 1 调
  - Parallel: 1 用户问→ N 调
  - Multiple+Parallel: N 函数 N 调
  - Relevance: 没合适函数 → 不应调
- 评判：**函数名 + 参数 exact match**

## 例题

```
User: "What's the weather in Tokyo and the time in NYC?"

Tools:
  get_weather(city: str)
  get_time(timezone: str)

Expected call:
  [
    {"name": "get_weather", "args": {"city": "Tokyo"}},
    {"name": "get_time", "args": {"timezone": "America/New_York"}}
  ]
```

## 评测维度

1. **AST match**（function name + args 全对）
2. **Executable check**（args 类型正确 + 函数实际可执行）
3. **Relevance**（"hello world" → 不调任何 tool）

## 分数 (BFCL v3, 2025.01)

| 模型 | Overall |
|------|---------|
| GPT-3.5-turbo | 60% |
| GPT-4o | 79% |
| Claude 3.5 Sonnet | 81% |
| Claude 3.7 Sonnet | 84% |
| **Functionary-V3** | **87%** |
| **gorilla-openfunctions-v3** | **88%** |

注：小专门模型（Functionary 7B）超过 GPT-4o，因为是 task-tuned。

## 工具调用格式

各家不同：
- **OpenAI** function call: `{"name": ..., "arguments": "..."}`（args 是 JSON string）
- **Anthropic** tool_use: structured object
- **LangChain ReAct**: `Action: name\nAction Input: ...`

→ BFCL 标准化成 OpenAI 格式，再 normalize 评测。

## 实操

src/bfcl_runner.py 3 题（weather / email / search）：

```python
from bfcl_runner import run_bfcl
from common import make_mock_model
import json

m = make_mock_model({"bfcl_1": '{"name":"get_weather","args":{"city":"Tokyo"}}'})
rs = run_bfcl(m)
print(rs[0]["passed"])  # True
```

## 一句话

> BFCL = "LLM 用工具的标准化考试"。
