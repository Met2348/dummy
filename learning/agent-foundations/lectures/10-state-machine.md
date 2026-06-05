# L10 · Agent as State Machine

## 30 秒核心

> Agent = **显式 state machine**：nodes (步骤) + edges (转移条件) + state (memory) = 流程可见、可控、可中断。

LangGraph (2024) 把这个思想做成事实标准。

## State machine vs ReAct loop

| 维度 | ReAct loop | State machine |
|------|-----------|---------------|
| 流程 | 隐式（thought 决定下步）| 显式 graph |
| 调试 | 看 trace | 看 graph |
| 中断 | 难 | edge 可加 interrupt |
| 持久化 | 全 trace | 只存 state |
| HITL | 难 | edge 自然 hook |

## LangGraph 核心抽象

```python
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("review", review_node)

graph.add_edge("plan", "execute")
graph.add_conditional_edges(
    "execute",
    lambda s: "review" if s.done else "execute",
)
graph.add_edge("review", END)
```

## State

```python
class AgentState(TypedDict):
    messages: list[dict]
    tool_calls: list[dict]
    plan: list[str]
    step: int
    done: bool
```

Reducer 函数控制 state 怎么合并（append vs replace）。

## 模式 1：Linear

```
START → plan → execute → END
```

## 模式 2：Loop with exit

```
START → think → tool → check ──┐
            ↑                   │
            └───── loop ────────┘
        (exit when check passes)
```

## 模式 3：HITL（human-in-loop）

```
START → plan → [INTERRUPT: human approve] → execute → END
```

LangGraph 提供 `interrupt()` + checkpoint 实现暂停/恢复。

## 模式 4：Multi-agent supervisor

```
START → supervisor ──→ worker_A ──┐
              ↓        worker_B   │
              ↓        worker_C   │
              └────── merge ←─────┘
                       ↓
                      END
```

## Checkpointing（持久化）

```python
from langgraph.checkpoint.sqlite import SqliteSaver
graph.compile(checkpointer=SqliteSaver(":memory:"))
```

- 每个 node 完成后存 state
- 失败可 resume
- HITL 必需（暂停时存）

## 我们手写版（`state_machine.py` 预告）

```python
class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}  # {(from,cond): to}
    def add_node(self, name, fn): ...
    def add_edge(self, src, dst, cond=None): ...
    def run(self, init_state, max_steps=10): ...
```

## 退出条件

- 能讲 state machine 4 模式
- 能写一个简单 StateGraph
- 知道 HITL 需要 checkpoint

## 一句话

> Agent 是显式 state machine —— LangGraph 让流程可见可控可中断，比隐式 ReAct loop 强一个数量级。
