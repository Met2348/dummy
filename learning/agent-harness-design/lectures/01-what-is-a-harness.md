# L01 · 什么是 Agent Harness

## 30 秒定位

把一个 raw model 变成能干活的 agent,中间那层**运行时引擎**就是 harness。

```
   ┌─────────── Harness(本专题造的东西)───────────┐
   │  system prompt · agentic loop · tool dispatch   │
用户→│  context mgmt · permissions · memory · tracing  │→ 干活
   │              ↕                                   │
   └──────────── raw model(LLM)───────────────────┘
```

Claude Code、Cursor、Devin、Codex CLI、OpenAI Agents SDK——**都是 harness**。它们包的可能是同一个模型,但能力天差地别,差在 harness。

## "harness 和 model 一样重要"

SWE-bench / Terminal-Bench 的公开结果反复说明同一件事:

> **换 harness 不换模型,分数能差几十分。**

同一个模型,一个 harness 给它干净的工具、好的 context 管理、能从错误里恢复;另一个给它一坨噪声、工具描述含糊、错误被吞——结果不可同日而语。本专题就是把这层引擎**亲手拆开造一遍**。

## 本专题 vs 上一专题(design patterns)

| | agent-design-patterns(L8) | agent-harness-design(本) |
|---|--------------------------|--------------------------|
| 层 | 设计层 | 运行时层 |
| 问 | 该不该造、造什么形状 | 用什么引擎跑 |
| 产物 | 5 workflow 模式选型 | 一个能跑的 mini-harness |

## mini-harness 的组件地图

本专题 [src/harness/](../src/harness/) 每个文件一个关注点:

| 组件 | 文件 | Lecture |
|------|------|---------|
| agentic loop | `loop.py` | L02 |
| tool 执行 | `tools.py` | L03-04 |
| context 管理 | `context.py` | L05-06 |
| sub-agent | `subagents.py` | L07 |
| 持久记忆 | `memory.py` | L08 |
| 权限 | `permissions.py` | L09 |
| system prompt | `system_prompt.py` | L10 |
| 错误恢复 | `errors.py` | L12 |
| 可观测 | `tracing.py` | L13 |
| 模型边界 | `model.py` | (贯穿) |
| 组装 | `mini_harness.py` | L16 |

L11(streaming/steering)、L14(hooks)、L15(评测 harness)是横切主题。

## 退出条件
- [ ] 能用一句话说清 harness 是什么、包在哪
- [ ] 理解"harness 和 model 一样重要"的实证依据
- [ ] 记住 mini-harness 的组件地图
