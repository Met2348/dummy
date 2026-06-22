# L07 · 工具 / MCP 控制平面

> Part II · 40-min lecture · 目标: 把「工具」从零散的函数调用, 升级成一个统一的**控制平面 (control plane)**——schema 校验、MCP 网关、多 agent 互联的单一治理点。

---

## 0. 工具派发注册表：一切能力的插入点

harness 的本构要素 ② (tool interface) 在工程上落成一个**工具派发注册表 (tool dispatch registry)**:

> **工具派发注册表是 MCP server 和 bash 共同插入的地方。** 不管能力来自一个 MCP server、一条 shell 命令、一个内部 API、还是一个 Python 函数, 它们都在这里**注册**, 模型通过统一的 schema 看到它们, harness 把模型的调用**路由**到对应实现。

你 `agent-harness-design/src/harness/tools.py` 的 `ToolRegistry`、本专题 `long_horizon.ToolRegistry` 都是它的最小版:

```python
reg.register("do_step", fn)        # 任何能力都在此注册
reg.specs()                        # 给模型看的 schema (只有名字/参数)
reg.dispatch(name, args, state)    # 把模型的请求路由到实现, 拿回结构化结果
```

关键: 模型只看到 **schema (接口)**, 看不到实现。这让能力可插拔——加一个 MCP server, 就是往注册表里多注册一批工具, harness loop 一行不改。

---

## 1. MCP：工具的「通用接口标准」

> **MCP (Model Context Protocol)**: 一个让工具/数据源以**标准化方式**暴露给 agent 的协议。类比: MCP 之于 agent 工具, 像 USB 之于外设——统一了「怎么插」。

有了 MCP, 工具生态从「每个 agent 自己写一套工具适配」变成「工具方实现一次 MCP server, 所有 agent 都能用」。在 harness 里, MCP server 就是注册表的一类**插件来源**。

```
                    ┌──────────────── harness ────────────────┐
   bash 命令 ──┐    │  ToolRegistry (统一 schema + 路由)         │
   内部 API ──┼───►│                                          │
   MCP server ┘    │  模型只看 schema, harness 负责 dispatch    │
                    └──────────────────────────────────────────┘
```

---

## 2. control plane：把网关合一

2026 出现了把多种「网关」合并成单一**控制平面**的趋势 (代表: `agentgateway` 这类开源 agentic proxy):

```
        ┌──────────────── Agent Control Plane ────────────────┐
        │                                                      │
        │   LLM gateway      MCP gateway       A2A gateway      │
        │   (agent↔模型)     (agent↔工具)      (agent↔agent)    │
        │                                                      │
        │   统一提供: 安全 · 可观测 · 治理 (drop-in)             │
        └──────────────────────────────────────────────────────┘
```

- **LLM gateway**: 统一管理 agent 到各模型的连接 (多 provider 路由、限流、计费)——就是 L03 provider 抽象的网络层版本。
- **MCP gateway**: 统一管理 agent 到工具的连接。
- **A2A gateway (agent-to-agent)**: 统一管理 agent 之间的通信 (多 agent 系统)。

把三者合一的价值: **安全、可观测、治理只需在一个地方做**, 而不是散落在每个 agent、每个工具适配里。这是企业级 harness 的「神经中枢」。

---

## 3. 工具设计的工程纪律（复用 + 强化）

你 Module 7 L04 讲过工具设计原则, 这里强化几条**生产级**要点:
- **schema 校验是控制点**: 模型给的参数可能乱来 (类型错、缺字段、注入)。注册表 dispatch 前必须**校验 schema** (本构要素 ④ control 的一部分)。schema 错配是 65% harness 失败的三大根因之一。
- **结构化结果, 而非裸字符串**: 工具回传应是结构化的 (成功/失败、数据、元信息), 便于 harness 判断和模型理解。
- **工具粒度**: 太细 (一堆原子工具) → 模型要编排很多步; 太粗 (一个万能工具) → 模型难用对。粒度本身是设计决策。
- **危险工具走权限门**: 写文件、执行 shell、调外部 API 这类**有副作用**的工具, 必须经过 L08 的权限/审批, 不能裸跑。

---

## 4. 本讲小结 + 通往 L08

- 工具派发注册表 = 一切能力 (bash / API / MCP) 的统一插入点; 模型只看 schema。
- MCP 是工具的「USB 标准」; control plane 把 LLM/MCP/A2A 三网关合一, 统一安全可观测治理。
- 生产纪律: schema 校验 (防 65% 根因之一)、结构化结果、合理粒度、危险工具走权限门。

> **下一讲 L08**: 控制机制 (本构要素 ④) 的正面战场——权限门、destructive-action hooks、预算护栏、企业级 RBAC/审批。这是「Model proposes, harness disposes」里 **disposes** 的全部细节, 也是把 agent 从「危险黑箱」变成「可托付系统」的关键。
