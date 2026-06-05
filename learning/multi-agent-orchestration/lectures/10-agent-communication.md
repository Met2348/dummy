# L10 · Agent 通信 — Message Bus / Pub-Sub

## 4 种通信模式

| 模式 | 描述 | 适用 |
|------|------|------|
| Direct | A → B → A | 1-to-1 |
| Broadcast | A → all | 公告 |
| Pub-Sub | A publish "topic X", B/C/D subscribed | event-driven |
| Shared blackboard | 所有 agent 读写 shared state | LangGraph reducer |

## Pub-Sub 模型

```
Agent A publish "search_done" → topic
                                  ↓
Agent B / C / D (subscribed) → callback
```

```python
class MessageBus:
    def __init__(self):
        self.subs = defaultdict(list)

    def subscribe(self, topic, callback):
        self.subs[topic].append(callback)

    def publish(self, topic, payload):
        for cb in self.subs[topic]:
            cb(payload)
```

## Shared blackboard（LangGraph 风格）

```python
class State(TypedDict):
    messages: Annotated[list, add]
    findings: Annotated[dict, lambda x,y: {**x,**y}]

# Node A
def researcher(state): return {"findings": {"r": "..."}}

# Node B
def coder(state): return {"findings": {"c": "..."}}

# Reducer 合并 findings
```

## A2A 风格通信

L06 (Topic 3) 讲过：通过 task lifecycle 通信，A → B 发 task，B 返 result。

## 消息格式标准

```python
@dataclass
class Message:
    from_agent: str
    to_agent: str | None  # None = broadcast
    topic: str
    payload: dict
    timestamp: float
    correlation_id: str  # tie req/resp
```

## 异步 vs 同步

| 模式 | 用 |
|------|---|
| Sync | 简单 routing |
| Async | streaming / parallel worker |
| Queue | 失败重试 |
| Event-driven | 大规模 |

## 实战：多 worker 并行

```python
import asyncio

async def run_workers_parallel(supervisor, query):
    workers = supervisor.select_parallel_workers(query)
    tasks = [asyncio.create_task(w.execute(query)) for w in workers]
    results = await asyncio.gather(*tasks)
    return supervisor.synthesize(results)
```

## 实现 (`message_bus.py` 预告)

```python
class MessageBus:
    def __init__(self):
        self.subs = {}
        self.history = []

    def subscribe(self, topic, agent_name, cb):
        self.subs.setdefault(topic, []).append((agent_name, cb))

    def publish(self, topic, payload, from_agent="?"):
        msg = {"topic":topic,"from":from_agent,"payload":payload}
        self.history.append(msg)
        for name, cb in self.subs.get(topic, []):
            cb(msg)
```

## 退出条件

- 能列 4 通信模式
- 能写 pub-sub
- 知道 LangGraph reducer 是 shared blackboard

## 一句话

> Agent 通信 4 模式 (direct/broadcast/pub-sub/blackboard) — pub-sub 灵活，blackboard 简单。
