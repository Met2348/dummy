# L05 · Pydantic AI（2024）

## 30 秒核心

> Pydantic AI = **type-safe agent**. 用 Pydantic schema 强制 LLM 输出 + tool args。

2024 末 Pydantic 团队推出（同 BaseModel 团队）。

## 设计哲学

> "FastAPI for AI"
> "LLM should be typed, validated, retried."

## Hello world

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class Weather(BaseModel):
    temperature: float
    conditions: str
    humidity: int

agent = Agent(
    "anthropic:claude-sonnet-4",
    result_type=Weather,
    system_prompt="You are a weather expert.",
)

result = agent.run_sync("Weather in Tokyo now")
print(result.data)  # Weather(temperature=22.5, conditions='sunny', humidity=65)
```

→ LLM 输出强制为 Weather BaseModel，不通过自动 retry。

## Tools as typed functions

```python
@agent.tool
def get_weather(ctx: RunContext, city: str) -> dict:
    """Get current weather for city."""
    return {"temp": 22, "humidity": 65}
```

`ctx: RunContext` 自动注入 deps / messages / etc。

## Dependency injection

```python
@dataclass
class Deps:
    db: Database
    api_key: str

agent = Agent("claude-sonnet-4", deps_type=Deps)

@agent.tool
async def query(ctx: RunContext[Deps], sql: str) -> list:
    return ctx.deps.db.query(sql)

result = await agent.run("...", deps=Deps(db=db, api_key="..."))
```

类 FastAPI dependency injection。

## 与 OpenAI SDK structured output

OpenAI 2024.08 推出 structured output (JSON Schema)，Pydantic AI 内化此：

```python
# OpenAI: 手写 schema
client.chat.completions.create(
    response_format={"type":"json_schema","json_schema":{...}}
)

# Pydantic AI: BaseModel 自动转 schema
agent = Agent(..., result_type=MyModel)
```

## Eval / streaming

```python
async with agent.run_stream("...") as stream:
    async for chunk in stream:
        print(chunk)
```

## 强弱

| 强 | 弱 |
|----|----|
| Type safety | 新，社区小 |
| FastAPI 老用户友好 | abstraction 同 LangChain 重 |
| Retry on validation fail | LLM 依赖支持 structured output |
| Multi-vendor | 不支持所有 LLM |

## 实战场景

| 场景 | 推荐 |
|------|------|
| 结构化输出严 | ✓ |
| Type-safe 团队 | ✓ |
| 数据 pipeline | ✓ |
| 自由文本 | LangChain 更松 |
| Multi-step agent | LangGraph 更强 |

## 我们 mock 版（`pydantic_ai_style.py` 预告）

简化：
- TypedAgent 接 result_type
- 模拟 LLM 输出 + JSON parse + retry on validation fail

## 退出条件

- 能默写 result_type 用法
- 能写 typed tool
- 知道 FastAPI 类比

## 一句话

> Pydantic AI = FastAPI for AI — Pydantic schema 强制 LLM 输出 + tool args，type-safe 团队首选。
