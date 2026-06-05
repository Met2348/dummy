# L02 · AutoGen（Microsoft 2023.10）

## 30 秒核心

> AutoGen = **ConversableAgent** 互相聊天，groupchat 拼出复杂工作流。

NeurIPS 2024，Microsoft 主推 multi-agent 框架之一。2024 AutoGen 0.4 重写为事件驱动。

## 核心抽象

| 抽象 | 含义 |
|------|------|
| `ConversableAgent` | 能 send/receive message 的 agent |
| `AssistantAgent` | LLM-based |
| `UserProxyAgent` | 代替人，可执行 code |
| `GroupChat` | N agent 轮流发言 |
| `GroupChatManager` | 决定下个发言人 |

## 简单 2-agent 例

```python
import autogen

user = autogen.UserProxyAgent("user", human_input_mode="NEVER",
                              code_execution_config={"work_dir":"out"})
assistant = autogen.AssistantAgent("assistant", llm_config={"model":"gpt-4"})

user.initiate_chat(assistant, message="Plot a sine wave to sine.png")
# assistant 写 code → user 自动 exec → assistant 看结果迭代
```

## GroupChat 例

```python
researcher = autogen.AssistantAgent("researcher", system_message="...")
writer = autogen.AssistantAgent("writer", system_message="...")
critic = autogen.AssistantAgent("critic", system_message="...")

group = autogen.GroupChat(
    agents=[researcher, writer, critic],
    messages=[],
    max_round=10,
)
manager = autogen.GroupChatManager(groupchat=group, llm_config={...})
user.initiate_chat(manager, message="Write blog about LLM 2025")
```

→ Manager 用 LLM 决"下个谁说"，agents 轮流贡献。

## 实现要点（`autogen_mock.py` 预告）

```python
class ConversableAgent:
    def __init__(self, name, system_message, reply_fn):
        self.name = name
        self.system = system_message
        self.reply_fn = reply_fn  # (history) -> reply

    def receive(self, message):
        self.history.append(message)
        reply = self.reply_fn(self.history)
        return reply
```

## AutoGen 强项

| 强 | 解释 |
|----|------|
| Code execution | UserProxyAgent 内置 exec |
| Flexible roles | role 完全自由配 |
| Research-friendly | 多 round 实验方便 |

## AutoGen 弱项

| 弱 | 解释 |
|----|------|
| 0.2 → 0.4 API breaking | 学习曲线陡 |
| Token 重 | groupchat 每 agent 看全 history |
| 调试难 | manager 路由不可见 |

## 退出条件

- 能默写 ConversableAgent / GroupChat / Manager 三抽象
- 知道 UserProxyAgent 能执行代码
- 知道 manager 用 LLM 决发言人

## 一句话

> AutoGen = ConversableAgent 互聊 + GroupChat manager 路由 — Microsoft 研究友好型多 agent 框架。
