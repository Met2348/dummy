# L03 · LangGraph Deep ⭐⭐⭐⭐⭐

## Topic 4 讲过基础，这里深化

## Persistence layer

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 3 backend 选
# 1. MemorySaver (dev)
# 2. SqliteSaver (single-user)
# 3. PostgresSaver (production)

with PostgresSaver.from_conn_string(DB) as saver:
    saver.setup()
    app = graph.compile(checkpointer=saver)
```

每 node 完成后写 checkpoint → state restore + HITL 必需。

## interrupt() — HITL

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    if state["amount"] > 10000:
        decision = interrupt({"reason": "approve transfer?", "amount": state["amount"]})
        return {"approved": decision}
    return {"approved": True}

# Frontend:
config = {"configurable": {"thread_id": "user_42"}}
result = app.invoke({"amount": 50000}, config)
# result.interrupted = True
# 用户决定 → resume
app.invoke(Command(resume="yes"), config)
```

## Sub-graphs

```python
sub = StateGraph(SubState)
sub.add_node("a", ...)
sub_app = sub.compile()

main = StateGraph(MainState)
main.add_node("sub_step", sub_app)  # 把 sub 当 node
```

## Time travel

```python
# 看历史 state
states = list(app.get_state_history(config))

# Resume from past state
app.invoke(None, {"configurable": {"thread_id":"...", "checkpoint_id": states[3].id}})
```

→ debugging 利器，可"倒带"重新试。

## React from langgraph

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[search, calc])
result = agent.invoke({"messages": [{"role": "user", "content": "Weather + 2+3"}]})
```

`create_react_agent` 是一键 ReAct agent 工厂。

## Streaming

```python
for chunk in app.stream(input, config, stream_mode="updates"):
    print(chunk)
# stream_mode:
#   - "updates" (per-node update)
#   - "values" (full state after each node)
#   - "debug" (everything)
#   - "messages" (LLM tokens)
```

## LangSmith 集成

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."

# 所有 graph 调用自动上 trace
```

LangSmith 是 LangChain 商业产品，trace + eval + dataset。

## LangGraph Cloud / Platform (2024)

- 部署 graph as service
- 自动 checkpoint + scale
- 加管理 dashboard
- Beta → 2025 GA

## 我们 mock 版（`langgraph_style.py` 预告）

延续 Topic 4 mock，加：
- `interrupt()` 函数
- Sub-graph as node
- Time-travel via history

## 退出条件

- 知道 3 checkpoint backend
- 能写 interrupt() HITL
- 知道 sub-graph + time travel + create_react_agent

## 一句话

> LangGraph 是 LangChain 的 2025 主战场 — checkpoint + interrupt + sub-graph + time travel —— production multi-agent 首选。
