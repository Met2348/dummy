# L02 · harness 的本构定义：4 要素 inclusion test

> Part I · 40-min lecture · 目标: 给「harness」一个**严格定义**, 而不是一个模糊的 buzzword。用它能鉴定一个系统到底是不是 harness, 把「生成器 / 护栏 / 工具包装」和「真 harness」分开。

---

## 0. 为什么需要本构定义

2026 年 "harness" 被滥用——套个 LLM 调用就敢叫 harness。但学界已经收敛出一个**本构定义** (constitutive definition): 一个 harness 当且仅当具备 **4 个必要充分要素**。

> **本构定义 (constitutive)**: 不是举例说明「harness 长什么样」, 而是给出「具备哪些要素就**算**、缺一个就**不算**」的判定标准。类比: 「水 = H₂O」是本构定义, 「水是透明的液体」只是描述 (酒精也符合)。

这 4 个要素被用作一个 **inclusion test (纳入测试)**, 套到真实系统上 (Claude Code / Codex CLI / Aider / Cline / OpenHands / SWE-agent), 干净地把它们和「不是 harness 的东西」分开。

---

## 1. 4 个必要充分要素

```
            ┌──────────────────────────────────────────────┐
            │              一个真 harness                   │
            │                                               │
            │   ① Agent Loop        ② Tool Interface        │
            │   (反复调用+决定何时停)  (暴露能力+路由调用)      │
            │                                               │
            │   ③ Context Mgmt      ④ Control Mechanisms    │
            │   (装什么进上下文)     (权限/预算/安全/防失控)    │
            └──────────────────────────────────────────────┘
              缺任意一个 → 不是 harness, 是别的东西
```

### ① Agent Loop（智能体循环）
反复地: 调用模型 → 解析输出 → 执行 tool call → 把结果喂回 → **决定是否继续**。
- 关键词是「**决定何时停**」。没有停止判断的, 不是 loop, 是单次调用。
- 这是心脏。70 系统实证研究里, **60% 的开源 agent 采用 Agent Loop 模式** (L11 详谈五大模式)。

### ② Tool Interface（工具接口）
把能力 (搜索 / 代码执行 / API / MCP server) 暴露给模型, 并把模型的请求**路由**到对应实现, 拿回结构化结果。
- 关键是「接口」: 模型只看到工具的 schema (名字、参数), 不关心实现。MCP、bash、内部 API 都从这里**插入**。

### ③ Context Management（上下文管理）
决定每一步**把什么放进上下文**: system prompt、历史、检索到的文档、中间产物、state; 窗口满了就 compaction / 摘要。
- 这就是 **context engineering**——prompt engineering 的升维版 (L04 深谈)。
- 长任务里, 这一项常常是性能瓶颈, 即使底层模型不变。

### ④ Control Mechanisms（控制机制）
在执行模型提议的动作前, 做**校验、授权、预算、安全**: schema 校验、权限门、花费上限、防失控 loop、destructive-action 拦截。
- 这是「Model proposes, **harness disposes**」里 dispose 的部分 (L08 深谈)。
- 没有它, agent 就是个黑箱: 漏 token、无限循环、执行危险命令。

---

## 2. inclusion test：拿它鉴定真实系统

把 4 要素当一张判定表, 逐个系统打勾：

| 系统 | ① loop | ② tools | ③ context | ④ control | 判定 |
|---|:--:|:--:|:--:|:--:|---|
| Claude Code | ✓ | ✓ | ✓ (5 阶段 compaction) | ✓ (权限+hooks) | **真 harness** |
| Codex CLI | ✓ | ✓ | ✓ | ✓ | **真 harness** |
| Aider / Cline / OpenHands / SWE-agent | ✓ | ✓ | ✓ | ✓ | **真 harness** |
| 一个只会 function-calling 的脚本 | ✗ (单次) | ✓ | ✗ | ✗ | ✗ 工具包装 |
| 一个 prompt 模板库 | ✗ | ✗ | 部分 | ✗ | ✗ 生成器 |
| 一个只做输入输出过滤的中间件 | ✗ | ✗ | ✗ | ✓ | ✗ 护栏(guardrail) |

> 用法对你: 以后读任何「我做了个 agent 框架」的论文/repo (这正是 `critical-reading-gap` 的攻击式阅读), 先用这张表过一遍——它到底是真 harness, 还是只实现了 1-2 个要素却号称 harness? 这是一个极快的 BS 探测器。

---

## 3. 一个关键架构观：harness 是确定性层

```
        ┌────────────────────────────┐
        │   模型 (随机, 会幻觉)        │   "提议": 我想调 rm -rf, 我想读这个文件
        │   stateless token predictor │
        └─────────────┬──────────────┘
                      │ proposes
        ┌─────────────▼──────────────┐
        │   harness (确定性运行时)     │   "处置": 校验 schema → 查权限 → 看预算
        │   validate/authorize/       │            → 安全规则 → 执行 → 记录
        │   execute/log               │
        └─────────────┬──────────────┘
                      │ acts on
        ┌─────────────▼──────────────┐
        │   真实世界 (文件/网络/shell) │
        └────────────────────────────┘
```

这个分层的价值: **把不确定性 (模型) 关在一个确定性的笼子里。** 模型可以天马行空地「提议」, 但每个动作都要过 harness 这道确定性闸门才能落到真实世界。这就是为什么有了 harness 才有**可预测性、安全性、可观测性**。

> 一个深刻的推论 (L12 会展开): 既然 harness 是确定性逻辑, 它理应是一个**清晰、可移植的工件**。但现实里, harness 逻辑往往**散落**在 controller 代码、框架默认值、工具适配器、verifier 脚本里, 不成体系——这本身就是一个研究 gap。

---

## 4. 本讲小结 + 通往 Part II

- harness 的本构定义 = 4 必要充分要素: **loop / tool interface / context mgmt / control**。
- inclusion test: 用它鉴定真假 harness, 区分生成器/护栏/工具包装。
- 核心架构观: harness 是把随机模型关进笼子的**确定性层** (Model proposes, harness disposes)。

> **Part II 开始 (L03-L08)**: 我们逐个要素往深里挖、往生产级做。从 ① 的「接真模型」开始——你 Module 7 的 mini-harness 用的是 MockModel, 现在把它换成一个能接 Anthropic/OpenAI/开源/Mock 的 **provider 抽象层**, 而 harness 主体一行不改。
