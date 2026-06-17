# Module 7 扩展：Agent Design Patterns + Agent Harness Design 双专题规划

> 设计日期：2026-06-17
> 学习仓库：`c:\Workspace\dummy`
> 模板来源：已完成的 Module 7 七专题（agent-foundations … agent-graduation）
> 触发：用户「仓库能否增加一些 agent design 的内容？」→ 选定"新建专题"形态 → 追加"harness design 巨大部分"

---

## 0. 为什么是这两个专题（填的是什么真空白）

Module 7 现有 7 专题覆盖了 agent 的**机制层**（ReAct/Reflexion/Plan-Execute/Router/StateGraph）和**框架层**（LangGraph/AutoGen/CrewAI/Claude SDK…），但有两层缺失：

| 缺失层 | 回答的问题 | 本次新增专题 |
|--------|-----------|-------------|
| **设计层** | agent 该**设计成什么形状**？该不该造 agent？ | Topic 8 `agent-design-patterns` |
| **运行时层** | 用什么**引擎**把 model 变成 agent？ | Topic 9 `agent-harness-design` |

- 现有 `agent-foundations` = "单 agent 循环原理"；`multi-agent-orchestration` = "多 agent 协作"。
- design patterns = 横切两者的**选型与权衡**（workflow vs agent、5 大 workflow 模式）。
- harness design = 把 model 跑起来的**工程引擎**（agentic loop / tool dispatch / context mgr / permission / trace）。SWE-bench/Terminal-Bench 上"同模型、不同 harness 差几十分"就是这一层的价值。

设计原则继承全仓库：**stdlib-only / CPU-only / mock LLM（关键词/模板确定性）/ 每个 src 模块带 `_self_test()` / aggregate test 汇总 / 每专题 scaffold→src→lectures→notebooks→README→environment→commit→tag**。

---

## 1. Topic 8 — `agent-design-patterns`（Module 7 第 8 专题）

### 定位
"该不该造 agent、造成什么形状"的架构决策层。核心蓝本：Anthropic《Building Effective Agents》(2024.12) 的 workflow-vs-agent 区分 + 5 大 workflow 模式；外延 Andrew Ng 4 大 agentic 模式、12-Factor Agents、context engineering、反模式。

### Lectures（14 篇）

| # | 文件 | 主题 |
|---|------|------|
| 01 | overview | Agent design 是什么 / 为何需要设计层 / workflow vs agent |
| 02 | when-not-to-agent | 何时**不**造 agent / 复杂度阶梯（单次调用→workflow→agent）/ cost-latency-reliability 权衡 |
| 03 | augmented-llm | 原子构件：augmented LLM（retrieval+tools+memory） |
| 04 | prompt-chaining | Workflow 模式 1：顺序分解 + gate |
| 05 | routing | Workflow 模式 2：分类 → 专门化处理 |
| 06 | parallelization | Workflow 模式 3：sectioning + voting |
| 07 | orchestrator-workers | Workflow 模式 4：动态分解 + worker 派发 |
| 08 | evaluator-optimizer | Workflow 模式 5：生成-评估循环 |
| 09 | autonomous-agents | 真 agent：开放循环 + 环境反馈 + stopping condition / 何时升级 |
| 10 | context-engineering | context window 作为资源 / compaction / sub-agent / 工具结果裁剪 |
| 11 | agentic-patterns | Ng 4 模式（reflection/tool use/planning/multi-agent）+ CoALA + 12-Factor |
| 12 | failure-modes | 反模式：over-engineering / context rot / runaway loop / tool sprawl / 错误累积 |
| 13 | design-checklist | 设计决策树 + 生产 checklist（guardrails/observability/HITL/cost ceiling） |
| 14 | capstone-pattern-zoo | Capstone：同一任务用 5 workflow + 1 agent 实现，对照成本/可靠性/延迟 |

### src/（stdlib-only，确定性 mock）

```
src/
├── common.py              # MockLLM(关键词模板) + Tool + CostTracker + trace
├── patterns/
│   ├── prompt_chaining.py
│   ├── routing.py
│   ├── parallelization.py     # sectioning + voting
│   ├── orchestrator_workers.py
│   ├── evaluator_optimizer.py
│   └── autonomous_agent.py    # 开放循环 + stopping condition
├── context_engineering.py # context 预算 / compaction / 裁剪 demo
├── failure_modes.py       # 反模式 demo + 修复（loop guard / context rot / tool sprawl）
├── capstone/pattern_zoo.py# 同任务 6 路实现 + 对照表
└── tests/test_all.py      # 汇总 _self_test
```

### Capstone：Pattern Zoo
- 任务："把一段自由文本产品需求 → 结构化工单（title/priority/labels/acceptance）"。
- 用 prompt-chaining / routing / parallelization / orchestrator-workers / evaluator-optimizer / autonomous-agent 各做一遍。
- 输出对照表：每种模式的 LLM 调用数、估算 token、可靠性评分、适用场景。
- 退出：6 路全 PASS，对照表生成。

### git tag：`agent-design-patterns`

---

## 2. Topic 9 — `agent-harness-design`（Module 7 第 9 专题，"巨大部分"）

### 定位
把一个 raw model 变成能干活的 agent 的**运行时引擎工程**。Claude Code / Cursor / Devin / Codex CLI 都是 harness。Capstone 是从零搭一个能跑的 mini-harness。

### Lectures（16 篇）

| # | 文件 | 主题 |
|---|------|------|
| 01 | what-is-a-harness | 什么是 harness / "harness 和 model 一样重要" / 各家 harness 对照 |
| 02 | agentic-loop | 核心控制流：model→tool_use→tool_result→model / stopping conditions / turn limit |
| 03 | tool-execution | tool 执行层：schema / dispatch / 并行 tool calls / 结果格式化 / 错误回传 |
| 04 | tool-design | 工具设计原则：粒度 / 命名 / 描述 / token 经济 / agent-facing vs human-facing |
| 05 | context-management | context window 作为预算 / 占用核算 / 截断策略 |
| 06 | context-compaction | compaction 实战：对话压缩 / 保留 vs 丢弃 / handoff |
| 07 | subagents | sub-agent 派生：隔离 context / fan-out / 结果聚合 / 何时用 |
| 08 | memory-subsystem | 持久记忆：file-based memory / session state / 跨会话 |
| 09 | permission-system | 权限/审批：allow/deny/ask / sandbox / 危险操作确认 / permission modes |
| 10 | system-prompt | harness system prompt 工程：角色/工具说明/行为约束/environment 注入 |
| 11 | streaming-steering | 流式输出 / 中途打断（steering）/ 用户消息插入 |
| 12 | error-recovery | transient vs terminal / 重试 / 降级 / 防 runaway loop |
| 13 | observability | trace/span / cost tracking / token 计量 / replay |
| 14 | hooks-extensibility | hooks / 扩展点 / MCP 集成 / 自定义 tool 注册 |
| 15 | eval-the-harness | 评测 harness：Terminal-Bench / SWE-bench harness 影响 / A-B 对照 |
| 16 | capstone-mini-harness | Capstone：从零搭可跑 mini-harness 跑通多步任务 |

### src/（"巨大部分" — 一个能跑的 mini-harness）

```
src/
├── harness/
│   ├── model.py          # MockModel：确定性，发 tool_use blocks
│   ├── loop.py           # agentic loop 引擎 + stopping condition
│   ├── tools.py          # ToolRegistry + dispatch + 并行执行
│   ├── context.py        # context 预算管理 + compaction
│   ├── memory.py         # file-based 持久记忆
│   ├── permissions.py    # allow/deny/ask + modes
│   ├── subagents.py      # sub-agent 派生 + 聚合
│   ├── tracing.py        # trace/span + cost tracking
│   ├── errors.py         # 错误分类 + retry + loop guard
│   └── system_prompt.py  # system prompt builder
├── mini_harness.py       # 组装成可运行 harness
├── capstone/run_task.py  # 在 mock 多步任务上跑通
└── tests/test_all.py     # 汇总 _self_test
```

### Capstone：Mini-Harness
- 组装 loop+tools+context+permission+memory+trace，给 MockModel + 一组 mock tools（read_file/write_file/calc/search）。
- 跑一个多步 mock 任务（如"读配置→算预算→写报告"），观察：完整 trace、tool 调用序列、context 占用、cost、permission 拦截。
- 退出：任务完成，trace 完整，loop guard 生效，所有 `_self_test` PASS。

### git tag：`agent-harness-design`

---

## 3. 收尾

- 更新课程索引/portfolio：Module 7 7→9 专题，全仓库 46→**48 专题**。
- 全局验证：两专题 `python -m ... test_all` 全 PASS + 两 capstone 产出对照表/trace。
- 大画像微调："造 agent 产品"补强为"会做 agent 设计选型 + 会搭 agent harness 引擎"。

## 4. 验证命令

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-design-patterns/src/tests/test_all.py
python learning/agent-design-patterns/src/capstone/pattern_zoo.py
python learning/agent-harness-design/src/tests/test_all.py
python learning/agent-harness-design/src/capstone/run_task.py
```
