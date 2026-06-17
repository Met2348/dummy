# Topic 8: Agent Design Patterns（Agent 设计模式）

> Module 7「Agent 应用层」第 8 专题 · 14 lectures · ~13h
>
> 设计视角：在写任何 `while` 循环前——**该不该造 agent?造成什么形状?**
> 蓝本 Anthropic《Building Effective Agents》(2024.12)。

## 这专题填的空白

Module 7 前 7 专题讲 agent 的**机制层**(ReAct/StateGraph)和**框架层**(LangGraph/CrewAI)。本专题往上一层补**设计层**:5 大 workflow 模式 + workflow-vs-agent 决策 + context engineering + 反模式。核心命题:**模型固定时,架构本身就是成本/可靠性杠杆**。

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Agent design 是什么 / workflow vs agent | (intro) |
| L02 | 何时**不**该造 agent / 复杂度阶梯 | (lecture) |
| L03 | 原子构件:Augmented LLM | `common.py` |
| L04 | **Prompt Chaining** | `patterns/prompt_chaining.py` |
| L05 | **Routing** | `patterns/routing.py` |
| L06 | **Parallelization** (sectioning + voting) | `patterns/parallelization.py` |
| L07 | **Orchestrator-Workers** | `patterns/orchestrator_workers.py` |
| L08 | **Evaluator-Optimizer** | `patterns/evaluator_optimizer.py` |
| L09 | **Autonomous Agent** | `patterns/autonomous_agent.py` |
| L10 | Context Engineering | `context_engineering.py` |
| L11 | Agentic patterns (Ng 4) + 12-Factor | (lecture) |
| L12 | 反模式与失败模式 | `failure_modes.py` |
| L13 | 设计决策树 + 生产 checklist | (lecture) |
| L14 | **Capstone**: Pattern Zoo (一任务六设计) | `capstone/pattern_zoo.py` |

## 5+1 模式速查

| 模式 | 形状 | 适用 |
|------|------|------|
| Prompt Chaining | 固定顺序 + gate | 有序可枚举的子任务 |
| Routing | 分类 → 专门 handler | 输入分多个明显类别 |
| Parallelization | 独立并行 / 多次投票 | 独立多面 / 单点高可靠 |
| Orchestrator-Workers | 动态拆 + 派发 | 子任务数运行时才知 |
| Evaluator-Optimizer | 生成-评估循环 | 有清晰评判且迭代有用 |
| Autonomous Agent | 开放循环自选工具 | 轨迹不可预测(最贵) |

## Tags

- `agent-design-patterns` — Module 7 第 8 专题

## 环境

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-design-patterns/environment/verify_env.py
```

stdlib-only,无 GPU、无 API key、无第三方包。

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-design-patterns/src/tests/test_all.py
```

预期:`10/10 modules passed`。

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-design-patterns/src/capstone/pattern_zoo.py
```

预期:六种设计的成本对照表,六行全 `[PASS]`,routing 最省(2 调用)、agent 最贵(5 调用)。

## 关键文献

- Building Effective Agents (Anthropic, 2024.12) — 本专题主线
- Effective context engineering for AI agents (Anthropic, 2025)
- A practical guide to building agents (OpenAI, 2025)
- Andrew Ng — Agentic Design Patterns (4 模式)
- 12-Factor Agents (Dexter Horthy)
- CoALA: Cognitive Architectures for Language Agents

## 与 Module 7 其它专题的边界

| 专题 | 层 | 回答 |
|------|----|------|
| agent-foundations | 机制 | 单 agent 循环怎么实现 |
| multi-agent-orchestration | 机制 | 多 agent 怎么协作 |
| agent-framework-stack | 工具 | 用哪个框架 |
| **agent-design-patterns(本)** | **设计** | **该不该造、造什么形状** |
| **agent-harness-design(下一专题)** | **运行时** | **用什么引擎跑** |

## 一句话

> 模型固定时,架构是成本杠杆。从最简单的设计开始,被任务逼着才往复杂爬——这就是 agent design 的全部纪律。
