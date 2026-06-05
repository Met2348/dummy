# L09 · Hierarchical Pattern（层级范式）

## 30 秒核心

> 1 个 Supervisor agent 决调度 + N 个 Worker agent 执行。LangGraph / Magentic-One / CrewAI hierarchical 都是此模式。

## 经典形态

```
                   Supervisor
                  /    |    \
            Worker A  Worker B  Worker C
              ↓        ↓         ↓
            ____________________________
                       ↓
                    Final
```

## Supervisor 职责

- 看用户 query → 选下一个 worker
- 看 worker 返回 → 决继续 / 换 / 终止
- 最终 → 合成 final answer

## Supervisor prompt 模板

```
You are a routing supervisor. Given:
- User goal: {goal}
- Available workers: {worker_list}
- Conversation so far: {history}

Decide next action:
- {worker_name}: dispatch to that worker
- finish: produce final answer

Output JSON: {"next": "worker_name" | "finish", "instruction": "..."}
```

## LangGraph supervisor 实现

```python
def supervisor(state):
    decision = llm(format_supervisor_prompt(state))
    if decision["next"] == "finish":
        return {"next": END, "final": decision["instruction"]}
    return {"next": decision["next"], "instruction": decision["instruction"]}

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {"researcher":"researcher","coder":"coder",END:END}
)
```

## Hierarchical 嵌套（深层级）

```
Top Supervisor
   ├─ Team A Supervisor
   │   ├─ Worker A1
   │   └─ Worker A2
   └─ Team B Supervisor
       ├─ Worker B1
       └─ Worker B2
```

→ "公司组织树"，每层一个 supervisor。

## 何时层级嵌套

| 用 | 不用 |
|----|------|
| 任务可清晰分层 | 任务网状 |
| 5+ workers | 2-3 workers |
| 团队多专业 | 单专业 |

## 实现 (`hierarchical.py` 预告)

```python
class Supervisor:
    def __init__(self, workers, name="supervisor"):
        self.workers = workers
        self.name = name
        self.history = []

    def route(self, state) -> str:
        # mock: keyword match
        q = state.get("query", "")
        for w in self.workers:
            if any(kw in q.lower() for kw in w.keywords):
                return w.name
        return "FINISH"

    def run(self, query, max_steps=10):
        state = {"query": query, "history": []}
        for _ in range(max_steps):
            next_w = self.route(state)
            if next_w == "FINISH":
                return self._synthesize(state)
            worker = next(w for w in self.workers if w.name == next_w)
            result = worker.execute(state)
            state["history"].append({"worker":next_w, "result":result})
        return self._synthesize(state)
```

## 退出条件

- 能讲 supervisor 三职责
- 能写 routing prompt
- 知道何时嵌套层级

## 一句话

> Hierarchical = 1 supervisor + N worker — routing 简单清晰，层级可嵌套，2025 多 agent 主流。
