# L06 · A2A 协议（Google 2025）⭐⭐⭐⭐

## 30 秒核心

> A2A (Agent-to-Agent) = **agent 之间的互操作协议**，不同框架/不同 vendor 的 agent 可以互相调对方的 skill。

Google 2025.04 发布，对应 MCP 解决 "tool 互操作"，A2A 解决 "agent 互操作"。

## MCP vs A2A

| 维度 | MCP | A2A |
|------|-----|-----|
| 对象 | LLM ↔ tool | Agent ↔ agent |
| 单元 | tool / resource / prompt | skill / task |
| 状态 | stateless | task lifecycle |
| 主导 | Anthropic | Google |
| 互补 | ✓ | ✓ |

## 核心抽象

```
Agent Card  = agent 的"简历" (能力 / endpoint / auth)
Skill       = agent 能做的事 (类似 method)
Task        = 一次 skill 调用 (有 lifecycle: pending → running → completed/failed)
Message     = task 中的对话消息
```

## Agent Card 示例

```json
{
  "name": "research-agent",
  "version": "1.0",
  "description": "Conducts deep web research",
  "url": "https://agent.example.com/a2a",
  "skills": [
    {
      "id": "deep_research",
      "name": "Deep Research",
      "description": "Multi-step web research with citations",
      "tags": ["research", "search"],
      "examples": ["Research X topic"]
    }
  ],
  "authentication": {"schemes": ["bearer"]}
}
```

→ 像 OpenAPI schema 之于 REST。

## Task lifecycle

```
1. Client → POST /tasks/send {skill_id, message}
   → 创建 task, 返回 task_id, status="pending"

2. Server → 异步处理 skill

3. Client polls or SSE: GET /tasks/{id}
   → status="running" with partial output
   → status="completed" with final output

4. (可选) Client 中途 input: POST /tasks/{id}/messages
```

## 与 MCP 关系

```
Outer:  Agent  ←─ A2A ─→  Agent
         ↓                  ↓
        MCP                MCP
         ↓                  ↓
        Tool              Tool
```

→ A2A 是 inter-agent，MCP 是 agent-tool。

## 何时用 A2A

| 场景 | 推荐 A2A | 原因 |
|------|---------|------|
| 跨组织 agent | ✓ | 协议标准 |
| 跨 vendor (Claude ↔ Gemini agent) | ✓ | 互操作 |
| 同组织同框架 multi-agent | ✗ | 框架内通讯更便 |
| 单 agent + tools | ✗ | MCP 够 |

## 我们手写版（in-process，`a2a_minimal.py` 预告）

```python
class A2AAgent:
    def __init__(self, card):
        self.card = card
        self.skills = {}
        self.tasks = {}

    def add_skill(self, skill_id, func):
        self.skills[skill_id] = func

    def send_task(self, skill_id, message):
        task_id = f"t{len(self.tasks)+1}"
        self.tasks[task_id] = {"status":"pending","output":None}
        try:
            output = self.skills[skill_id](message)
            self.tasks[task_id] = {"status":"completed","output":output}
        except Exception as e:
            self.tasks[task_id] = {"status":"failed","error":str(e)}
        return task_id

    def get_task(self, task_id):
        return self.tasks.get(task_id)
```

## 退出条件

- 能讲 MCP vs A2A 分工
- 能默写 Agent Card 字段
- 知道 Task 5 状态 (pending/running/completed/failed/cancelled)

## 一句话

> A2A = agent 间的"互操作 OpenAPI" — Google 2025 主推，互补 MCP。
