# L04 · LangGraph ⭐⭐⭐⭐⭐

## 30 秒核心

> LangGraph = **StateGraph + Node + Edge + Checkpoint**，把 agent 流程做成显式状态机。

LangChain 2024 主推，2025 上半年生产部署 multi-agent 首选。

## 核心抽象

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]  # reducer: append
    step: int

graph = StateGraph(State)
graph.add_node("plan", plan_fn)
graph.add_node("execute", execute_fn)
graph.add_node("review", review_fn)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges(
    "execute",
    lambda s: "review" if s["step"] > 5 else "execute",
)
graph.add_edge("review", END)

app = graph.compile(checkpointer=MemorySaver())
result = app.invoke({"messages":[], "step":0}, config={"thread_id":"1"})
```

## 4 强项

| 强 | 解释 |
|----|------|
| **State explicit** | TypedDict + Annotated reducers |
| **Conditional edges** | LLM 或代码决下一节点 |
| **Checkpoint** | state persist + resume |
| **HITL** | `interrupt()` 中断等人 |

## Reducer 模式

```python
class State(TypedDict):
    messages: Annotated[list, add]      # append-only
    tool_calls: Annotated[list, lambda x,y: y]  # replace
    counter: Annotated[int, lambda x,y: x + y]  # sum
```

→ 多 node 同时更新同 field 时，reducer 决怎么合并。

## Multi-agent in LangGraph

```python
# Supervisor pattern
graph.add_node("supervisor", supervisor_fn)
graph.add_node("researcher", researcher_fn)
graph.add_node("coder", coder_fn)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],   # supervisor 决去哪
    {"researcher":"researcher","coder":"coder",END:END}
)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
```

## Checkpoint 与 resume

```python
config = {"thread_id":"user_42"}
app.invoke(input1, config)   # 跑到 interrupt
# ... 时间过去
app.invoke(input2, config)   # 从上次 state 继续
```

支持 SQLite / Postgres / Redis backend。

## HITL via interrupt

```python
def review_node(state):
    if needs_approval(state):
        interrupt({"reason":"approve action", "action":state.action})
    return state
```

→ Run 暂停 → frontend 取出 interrupt 信息 → 用户决定 → resume。

## 我们手写版（`langgraph_mock.py` 预告）

```python
class StateGraph:
    def __init__(self, state_type):
        self.nodes = {}
        self.edges = {}
        self.conditional = {}

    def add_node(self, name, fn): ...
    def add_edge(self, src, dst): ...
    def add_conditional_edges(self, src, fn): ...
    def compile(self): return CompiledGraph(self)
```

## 与 Module 6 state_machine.py 关系

- Module 6 (agent-foundations) 已经实现了一个简单版
- 本 lecture 加：reducer / checkpoint / interrupt 概念

## 退出条件

- 能默写 StateGraph / add_node / add_edge / add_conditional_edges
- 能解释 reducer 用途
- 知道 checkpoint + interrupt

## 一句话

> LangGraph = 显式 StateGraph + reducer + checkpoint + HITL — 2025 multi-agent 生产首选。
