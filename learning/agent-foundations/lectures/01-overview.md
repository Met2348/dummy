# L01 · Agent 是什么

## 30 秒定义

**Agent = LLM × (perception + action + memory) × loop**

一个 agent 不止是 LLM 调用，是一个**循环系统**：
- **perception**：感知环境（用户输入、tool 返回、外部 API）
- **action**：影响环境（调 tool、写文件、发请求）
- **memory**：跨 turn 记住状态
- **loop**：上面三者反复执行直到 done

## 与"普通 chatbot"区别

| 维度 | Chatbot | Agent |
|------|---------|-------|
| 调用 | LLM(prompt) → text | LLM + tool + memory loop |
| 状态 | 上下文 history | 显式 state machine |
| 副作用 | 无 | 可读写外部世界 |
| 多步 | 一问一答 | 多步规划 |
| 失败处理 | 用户重问 | 自 reflect / retry |

## Agent 五大组件

```
        ┌─────────────┐
input → │   Planner   │ → 拆步骤
        └─────┬───────┘
              ↓
        ┌─────────────┐
        │   Router    │ → 选 tool
        └─────┬───────┘
              ↓
        ┌─────────────┐
        │ Tool calls  │ → 调外部
        └─────┬───────┘
              ↓
        ┌─────────────┐
        │  Observer   │ → 观察结果
        └─────┬───────┘
              ↓
        ┌─────────────┐
        │   Memory    │ → 存中间
        └─────────────┘
```

## 2025-2026 agent 范式 5 大类

| 范式 | 代表 | 时间 |
|------|------|------|
| ReAct | Yao 2022 | thought-action-obs |
| Reflexion | Shinn 2023 | 失败后 reflect |
| Plan-and-Execute | LangChain | 先规划再执行 |
| Router | LangChain | LLM 当路由 |
| State machine | LangGraph 2024 | 显式 StateGraph |

后续 L02-L10 分别深入。

## 退出条件

- 能说出 chatbot vs agent 三大差异
- 能画 agent 五组件循环图
- 能列举 5 范式名字

## 一句话

> Agent = LLM + tool + memory + loop —— 不止"会聊"，还要"会做事 + 会记 + 会反思"。
