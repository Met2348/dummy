# Topic 1: Agent Foundations（Agent 基础范式）

> Module 7「Agent 应用层」第 1 专题 · 12 lectures · ~13h
>
> 工程视角：从 ReAct 到 LangGraph-style state machine，理解 agent 五大范式

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Agent 是什么 | (intro) |
| L02 | **ReAct** (Yao 2022) | `react_loop.py` |
| L03 | CoT / ToT / GoT | (lecture) |
| L04 | **Reflexion** | `reflexion_demo.py` |
| L05 | AutoGPT / BabyAGI | (lecture) |
| L06 | **Plan-and-Execute** | `plan_execute.py` |
| L07 | Router pattern | `router_pattern.py` |
| L08 | Tool 抽象基础 | `tools/` |
| L09 | System prompt patterns | (lecture) |
| L10 | Agent as state machine | `state_machine.py` |
| L11 | Agent 调试 | `tracing.py` |
| L12 | **Capstone**: 手写 ReAct loop | `capstone_react.py` |

## Tags

- `agent-foundations` — Module 7 第 1 专题

## 跑测试

```powershell
python learning/agent-foundations/src/tests/test_agent_foundations.py
```

预期：`all modules passed`。

## 跑 Capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/agent-foundations/src'); from capstone_react import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 关键文献

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao 2022)
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn 2023)
- LangGraph: stateful agent applications
- Plan-and-Solve Prompting (Wang 2023)
- Toolformer (Schick 2023)
- AutoGPT / BabyAGI (2023 玩具开源)

## 5 范式对照

| 范式 | 特征 | 适用 |
|------|------|------|
| ReAct | thought-action-obs loop | 通用，灵活 |
| Reflexion | 失败后 self-reflect | 多次试错 |
| Plan-and-Execute | 先全规划再执行 | 任务可预先分解 |
| Router | LLM 当路由分发 | 多 expert agent |
| State machine | StateGraph 明确节点 | 流程清晰可控 |

## 与 Module 6 / Module 4 区别

| Module | 视角 | 重点 |
|--------|------|------|
| Module 4 multimodal-agent | RL | 训练 VLM agent |
| Module 6 agent-code-eval | 评测 | 跑 SWE-Bench/WebArena |
| **Module 7 agent-foundations (本)** | **工程** | **手写 agent 范式** |

## 一句话

> 5 个 agent 范式 (ReAct / Reflexion / Plan-Execute / Router / State machine) — 用 stdlib 全部手写一遍，理解 agent 的"内核"。
