# L11 · 五大架构模式（70 系统实证）

> Part III · 40-min lecture · 目标: 给你一张「harness 架构地图」——70 个开源 agent 系统实证出的五种执行模式 + 一个统一调度框架, 让你按任务选对形态。

---

## 0. 你造的只是其中一种

你这套 mini-harness 是 **Agent Loop** 形态。但它不是唯一形态。一份 **2026-04 对 70 个公开 agent 系统的实证研究**, 沿五个维度 (subagent 架构 / context 管理 / 工具系统 / 安全机制 / 编排) 分析, 综合出**五种架构模式**, 并提出一个把它们统一起来的**调度框架**。

> 这正是你 `critical-reading-gap` 里「文献综述 + 领域地图」的产物形态: 不是读一篇, 而是把一个子领域的 70 个系统**归纳成模式**。把这一讲当成一张现成的领域地图。

---

## 1. 五大模式

```
① Agent Loop        ②  Event-driven      ③ State-machine
   反复 call→act→     由外部事件触发         显式状态 + 转移
   feedback→决定停     (webhook/消息/定时)    (确定性强)
   ★ 70 系统中 60%

④ Graph / Flow      ⑤ Hybrid
   DAG/图编排节点       上面几种的组合
   (LangGraph 式)       (生产系统常态)
```

| 模式 | 一句话 | 何时用 | 代价 |
|---|---|---|---|
| ① **Agent Loop** | 反复 call→act→feedback, 自己决定何时停 | 开放式任务、探索性、SWE | 难预测、难控 (要靠 L08 护栏) |
| ② **Event-driven** | 外部事件驱动 (消息到达/定时/webhook) | 长驻服务、被动响应 | 编排复杂 |
| ③ **State-machine** | 显式状态 + 合法转移 | 流程确定、合规要求高 | 不灵活、状态爆炸 |
| ④ **Graph/Flow** | 把工作画成 DAG, 节点是步骤/agent | 多步、可并行、需可视编排 | 图设计本身是负担 |
| ⑤ **Hybrid** | 组合上面几种 | 真实生产系统 | 复杂度 |

> **60% 的开源 agent 用 Agent Loop**——它是默认, 因为最灵活。但「灵活」的代价是「难控」, 所以 Agent Loop 系统对 L08 控制层的依赖最重。State-machine 反过来: 强可控但不灵活。**没有最优形态, 只有任务-形态匹配。**

---

## 2. 统一调度框架：它们其实是一个东西的不同投影

那份研究的深刻之处: 提出一个**统一控制模型**, 把五种模式都看成同一个调度器的不同配置:

```
        统一调度器 (scheduler)
        ┌────────────────────────────────────────┐
        │  待办 (what to do next)                  │
        │  ├ 选下一步: 模型自决?(Loop)              │
        │  │           事件触发?(Event)            │
        │  │           状态转移?(State-machine)    │
        │  │           图依赖?(Graph)              │
        │  ├ 执行 (tool dispatch — L07)            │
        │  ├ 更新状态 (context/state — L04/L05)    │
        │  └ 控制 (permission/budget — L08)        │
        └────────────────────────────────────────┘
```

看出来了吗——**这个统一调度器的四个部件, 正是 L02 的 4 本构要素** (loop/tool/context/control)。五种「模式」的差别, 只在「**怎么选下一步**」: 模型自决、事件触发、状态转移、还是图依赖。其余三件 (执行/状态/控制) 是共享的。

> 这是一个漂亮的统一视角, 也是一个**研究信号** (L13): 既然能统一建模, 为什么现实里 harness 逻辑还是散落、不可移植? (→ L12)

---

## 3. 怎么用这张地图选型

```
任务可预测、合规重    → State-machine / Graph
任务开放、要探索      → Agent Loop (+ 强 L08 护栏)
被动响应、长驻        → Event-driven
多步可并行、要可视     → Graph/Flow
真实生产             → Hybrid (按子任务混用)
```

你 Module 7 `agent-design-patterns` 讲过「该不该 agent、5 种 workflow 模式」, 那是**应用层**的选型 (workflow vs agent)。本讲是**运行时层**的选型 (用哪种调度形态实现)。两者配套: 先定 workflow 形状, 再选 harness 调度模式。

---

## 4. 本讲小结 + 通往 L12

- 五大模式: Agent Loop (60%) / Event-driven / State-machine / Graph-Flow / Hybrid。
- 统一调度框架: 五模式 = 同一调度器在「怎么选下一步」上的不同投影; 其余三件 (执行/状态/控制) = L02 的本构要素。
- 选型: 没有最优形态, 只有任务-形态匹配。

> **下一讲 L12**: 既然 harness 能被统一建模, 它理应是一个清晰、可移植的工件。但现实是 harness 逻辑**散落**在各处——这是 Part III 的最后一块, 也是一个真实的研究前沿: **portable harness** 与 *Natural-Language Agent Harnesses*。
