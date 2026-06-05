# L08 · Tool 抽象基础

## Tool 是什么

> Tool = 一个**有 schema、可调用、返回结构化结果**的函数。

从 agent 视角：tool 是 agent 接触外部世界的唯一通道。

## Tool 三件套

```python
@dataclass
class Tool:
    name: str        # "calculator"
    description: str # "Compute math expression"
    schema: dict     # {"expression": "string"}
    func: Callable   # actual implementation
```

LLM 看到 schema 就知道**何时**、**怎么**调。

## 5 类常见 tool

| 类别 | 例 | 注意 |
|------|---|------|
| Compute | calculator / python_exec | sandbox 必需 |
| Retrieve | search / RAG / DB query | rate-limit + cache |
| Read | file_read / api_get | size limit |
| Write | file_write / api_post | confirm / dry-run |
| Communicate | email / slack | HITL |

## Schema 标准化

```json
{
  "name": "search",
  "description": "Web search",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "..."}
    },
    "required": ["query"]
  }
}
```

OpenAI / Anthropic / Google 都用类似 JSON Schema。

## Tool 调用流程

```
LLM output: tool_call{name="search", args={"query":"X"}}
       ↓
Parser: 抽 name + args
       ↓
Validator: 校验 schema
       ↓
Executor: 调实际函数
       ↓
Result: 包装为 ToolResult{ok, value, error}
       ↓
Observation: 写回 prompt history
```

## Tool 设计 4 原则

| 原则 | 解释 |
|------|------|
| **Atomic** | 一个 tool 一件事，别 god-class |
| **Idempotent** | 可重复调 (read-only 尤其) |
| **Loud failure** | 错误必须明确返回，别静默 |
| **Self-described** | description + schema 自带文档 |

## 与 L08 后续（MCP/A2A）的关系

- 本 lecture：**tool 抽象**（实现细节）
- Topic 3 MCP：**tool 协议**（互操作）
- 关系：MCP 把 tool 抽象标准化为跨进程协议

## 实现核心（`tools/` 目录预告）

```python
# tools/calculator.py
TOOL = Tool(
    name="calculator",
    description="...",
    schema={"expression": "string"},
    func=lambda args: {"result": eval(args["expression"])}
)
```

实际 capstone 4 工具：calculator / search_mock / file_op / web_mock。

## 退出条件

- 能默写 Tool 三件套（name/description/schema/func）
- 知道 5 类 tool 各自风险
- 能列 4 设计原则

## 一句话

> Tool = agent 接触世界的钥匙 —— name + schema + func 三件套，atomic / idempotent / loud / self-described。
