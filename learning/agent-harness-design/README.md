# Topic 9: Agent Harness Design（Agent 运行时引擎)

> Module 7「Agent 应用层」第 9 专题 · 16 lectures · ~16h
>
> 运行时视角:把一个 raw model 变成能干活的 agent,中间那层**引擎**怎么造。
> Capstone 是从零搭一个能跑的 mini-harness。

## 这专题填的空白

design-patterns(第 8 专题)讲"该不该造、造什么形状";本专题讲"**用什么引擎跑**"。Claude Code / Cursor / Devin / Codex CLI 都是 harness——同模型、换 harness,SWE-bench 能差几十分。这层亲手拆开造一遍。

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 什么是 harness / "harness 和 model 一样重要" | (intro) |
| L02 | **Agentic Loop**(心脏) | `harness/loop.py` |
| L03 | Tool 执行层 | `harness/tools.py` |
| L04 | 工具设计原则 | `harness/tools.py` |
| L05 | Context 管理(窗口即预算) | `harness/context.py` |
| L06 | Context Compaction | `harness/context.py` |
| L07 | Sub-agents(隔离 context) | `harness/subagents.py` |
| L08 | Memory 子系统(持久记忆) | `harness/memory.py` |
| L09 | **权限/审批系统** | `harness/permissions.py` |
| L10 | System Prompt 工程 | `harness/system_prompt.py` |
| L11 | 流式 / 中途打断(steering) | (concept + 挂载点) |
| L12 | 错误恢复 / 防失控 | `harness/errors.py` |
| L13 | 可观测性(trace + cost) | `harness/tracing.py` |
| L14 | Hooks / 可扩展性 | (concept + 扩展点) |
| L15 | 评测 harness | (concept) |
| L16 | **Capstone**: Mini-Harness 跑通多步任务 | `mini_harness.py` + `capstone/run_task.py` |

## mini-harness 组件地图

```
mini_harness.Harness
 ├─ system_prompt.build_system_prompt   角色+工具+环境
 ├─ context.ContextWindow               窗口预算 + compaction
 ├─ tools.ToolRegistry                  注册 + dispatch + 结构化结果
 ├─ permissions.PermissionManager       auto / readonly / ask
 ├─ tracing.Trace + CostTracker         可观测 + 成本
 ├─ errors.LoopGuard + with_retry       防失控 + 重试
 ├─ model.MockModel                     确定性 brain(替身 LLM)
 └─ loop.run_loop                       ★ agentic loop:串起以上一切
        + subagents.run_subagent / fan_out   隔离派生
        + memory.Memory                       跨会话持久
```

## Tags

- `agent-harness-design` — Module 7 第 9 专题

## 环境

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-harness-design/environment/verify_env.py
```

stdlib-only,无 GPU、无 API key、无第三方包。

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-harness-design/src/tests/test_all.py
```

预期:`13/13 modules passed`。

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-harness-design/src/capstone/run_task.py
```

预期:同任务跑两遍——`ask` 模式写成功、`readonly` 模式写被拦并 surface;完整 agentic loop trace + cost。

## 关键文献

- Building Effective Agents / Building agents with the Claude Agent SDK (Anthropic)
- Effective context engineering for AI agents (Anthropic, 2025)
- 12-Factor Agents (Dexter Horthy) — own your prompts / context / control flow
- SWE-bench / Terminal-Bench — harness 对分数的影响
- A practical guide to building agents (OpenAI, 2025)

## 与 Module 7 其它专题的边界

| 专题 | 层 | 回答 |
|------|----|------|
| agent-foundations | 机制 | 单 agent 循环怎么实现 |
| agent-design-patterns | 设计 | 该不该造、造什么形状 |
| **agent-harness-design(本)** | **运行时** | **用什么引擎跑** |
| tool-use-mcp | 协议 | 工具调用的协议 |
| agent-memory-context | 算法 | 记忆的算法 |

## 一句话

> agent 能力 = model × harness。把 MockModel 换成真 LLM,这套 loop/tools/context/permission/memory/trace 的形状一模一样——你已经造过那台引擎了。
