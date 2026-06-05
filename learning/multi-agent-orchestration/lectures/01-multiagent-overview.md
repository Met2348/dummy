# L01 · 多 agent 三大范式

## 30 秒：为什么需要多 agent

| 单 agent 问题 | 多 agent 解 |
|--------------|------------|
| 长 prompt → role 混 | 分角色 |
| 专业度不够 | role-specialized agent |
| 长任务 fail | 拆任务 |
| 单视角偏见 | 多 agent debate |
| 一个 LLM context 装不下 | 多 agent 各管一段 |

## 三大范式

```
1. Hierarchical (层级)
       supervisor
         /  |  \
     worker  worker  worker

2. Debate (辩论)
       agent A → critique → agent B
                                ↓
       agent B → critique → agent A
                                ↓
                             judge → final

3. Collaborative (协作)
       agent A ←─→ agent B
            ↘   ↙
            agent C
            (shared memory / msg bus)
```

## 何时多 agent 反而坏

| 反模式 | 表现 |
|-------|------|
| Token 爆炸 | 5 agent 互聊 → 100× context |
| 无终止条件 | 互相礼让无人 commit |
| Round-robin 浪费 | 每 agent 都说话但只 1 个 relevant |
| 测试不可见 | trace 跨 agent 难看清 |
| 失败传染 | 1 agent hallucinate → 全 crew 被带跑 |

→ Anthropic 2024 blog "Building effective agents" 建议：**先单 agent + tool，多 agent 是 last resort**。

## 6 框架按范式分

| 框架 | 主范式 | 团队 |
|------|--------|------|
| AutoGen | 协作 (conversable agent) | Microsoft |
| CrewAI | 层级 (crew + role) | joaomdmoura |
| LangGraph | 状态机 | LangChain |
| MetaGPT | 层级 + SOP | DeepWisdom |
| Magentic-One | 层级 (Orchestrator + worker) | Microsoft |
| Swarm | 协作 (hand-off) | OpenAI |

## 终止条件 5 类

| 类 | 例 |
|----|---|
| Goal reached | judge says "done" |
| Step cap | max_rounds=10 |
| Token cap | total < 50k |
| Cost cap | < $5 |
| HITL | 人工 approve |

## 退出条件

- 能讲三大范式
- 知道何时不用多 agent
- 能配 6 框架到范式

## 一句话

> 多 agent = 拆角色 + 拆任务 + 加 debate — 但 token 也乘 N，先单 agent 起步。
