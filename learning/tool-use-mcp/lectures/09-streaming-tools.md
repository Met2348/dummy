# L09 · Streaming + Interrupt Tools

## 两类需 streaming 的工具

| 类 | 例 | 为什么 stream |
|----|---|--------------|
| 长执行 | LLM call / large download / long compute | 让用户看进度 |
| 实时数据 | log tail / sensor / chat msg | 不断更新 |

## Async generator pattern

```python
async def long_search(query: str):
    yield {"status": "fetching"}
    docs = await fetch_async(query)
    yield {"status": "ranking", "n_docs": len(docs)}
    ranked = rerank(docs, query)
    yield {"status": "done", "results": ranked}
```

→ LLM 看到每段 yield 都能更新 UI。

## Interrupt 模型

```
LLM → start_tool(id=abc)
       ↓
       tool emit partial yield 1
       tool emit partial yield 2
                                ← User: "stop"
                          ← LLM: send_interrupt(abc)
       tool catches CancelledError
       cleanup
```

## 实现 (`streaming_tools.py` 预告)

```python
class StreamingTool:
    def __init__(self):
        self.cancelled = False

    def start(self, params):
        self.cancelled = False
        for i in range(10):
            if self.cancelled:
                yield {"status":"cancelled"}; return
            yield {"step": i, "progress": (i+1)/10}
            time.sleep(0.05)
        yield {"status":"done"}

    def cancel(self):
        self.cancelled = True
```

## OpenAI streaming tools (2024.06)

```python
stream = client.chat.completions.create(
    ...,
    stream=True,
    tools=[...]
)
for chunk in stream:
    if delta := chunk.choices[0].delta.tool_calls:
        # 部分 tool call args
        ...
```

→ LLM 边 stream 边输出 tool_call args，用户可中途 cancel。

## Resumability

| 模式 | 重连 |
|------|------|
| Stateless | 全部重做 |
| Checkpoint | 从 last yield 恢复 |
| Cursor | server 记 offset |

## 应用场景

| 场景 | streaming 需求 |
|------|---------------|
| Deep research | progress + partial citation |
| Code interpreter | live stdout |
| Log analysis | tail incremental |
| Multi-step agent | each step yield |

## 退出条件

- 能写 async generator tool
- 知道 cancel 机制
- 了解 OpenAI streaming tool_calls

## 一句话

> Streaming tool = async generator 边产边吐 + cancel 钩子 — UX 让人能看进度并中断。
