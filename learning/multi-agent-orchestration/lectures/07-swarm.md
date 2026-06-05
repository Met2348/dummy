# L07 · OpenAI Swarm（2024.10）

## 30 秒核心

> Swarm = 极简 **hand-off** 模型 — agent 通过返回另一 agent 完成"控制移交"。

OpenAI 2024.10 实验性发布，~500 行 Python。"教育性框架"。

## 核心抽象

```python
from swarm import Agent, Swarm

def transfer_to_b():
    return agent_b  # hand-off!

agent_a = Agent(name="A", instructions="...", functions=[transfer_to_b])
agent_b = Agent(name="B", instructions="...")

client = Swarm()
response = client.run(agent=agent_a, messages=[{"role":"user","content":"hi"}])
print(response.agent.name)  # 可能是 "B"
```

## Hand-off 机制

```
Agent A 是 active
       ↓
LLM 选 transfer_to_b function
       ↓
function 返回 agent_b
       ↓
Swarm framework 把 active 换成 B
       ↓
后续 LLM call 用 B 的 instructions + functions
```

→ 比 "supervisor 决调度" 更简单：agent 自己决何时转给谁。

## Context 传递

```python
context = {"user_name": "Alice", "topic": "RAG"}
response = client.run(
    agent=agent_a,
    messages=msgs,
    context_variables=context,
)
```

`{user_name}` 在 instructions 里自动渲染。

## 与 LangGraph hand-off 对比

| 维度 | Swarm | LangGraph |
|------|-------|-----------|
| Hand-off 触发 | function 返 agent | conditional_edges |
| 状态 | context_variables (dict) | TypedDict + reducer |
| Checkpoint | 无 | 有 |
| HITL | 无 | interrupt() |
| 部署 | 教育，不推 | 生产首选 |

OpenAI 官方说 Swarm 不推生产用，是范式 demo。

## 实现 (`swarm_mock.py` 预告)

```python
@dataclass
class SwarmAgent:
    name: str
    instructions: str
    functions: list[Callable]

class Swarm:
    def run(self, agent, messages, max_turns=10):
        active = agent
        for _ in range(max_turns):
            reply = self._llm_call(active, messages)
            if hasattr(reply, "transfer_to"):
                active = reply.transfer_to  # hand-off!
                continue
            return {"agent": active.name, "messages": messages + [reply]}
```

## 何时用 Swarm 风格

| 用 | 不用 |
|----|------|
| 简单分诊 (triage) | 复杂状态 |
| 教学 demo | 生产 |
| Routing → expert | 长任务 |

## 现在状态（2025）

OpenAI 自己已经把 Swarm 思路融进 **OpenAI Agents SDK** (2025.04)，Swarm 停止更新。

## 退出条件

- 能讲 hand-off 机制
- 知道与 LangGraph 区别
- 知道 OpenAI Agents SDK 是后继

## 一句话

> Swarm = 极简 hand-off 范式 — function 返 agent 实现 routing，500 行 Python，教育用。
