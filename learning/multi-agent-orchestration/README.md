# Topic 4: Multi-Agent Orchestration（多 agent 编排）

> Module 7 第 4 专题 · 13 lectures · ~14h
>
> AutoGen / CrewAI / LangGraph / MetaGPT / Magentic-One / Swarm 6 大编排范式

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 多 agent 三大范式 | (intro) |
| L02 | **AutoGen** (MS 2023) | `autogen_mock.py` |
| L03 | **CrewAI** (2024) | `crewai_mock.py` |
| L04 | **LangGraph** ⭐ | `langgraph_mock.py` |
| L05 | **MetaGPT** (DeepWisdom 2023) | `metagpt_mock.py` |
| L06 | **Magentic-One** (MS 2024.11) | `magentic_one_mock.py` |
| L07 | **OpenAI Swarm** (2024.10) | `swarm_mock.py` |
| L08 | Debate pattern | `debate.py` |
| L09 | Hierarchical (supervisor + worker) | `hierarchical.py` |
| L10 | Agent 通信 (msg bus / pub-sub) | `message_bus.py` |
| L11 | 冲突 resolution (vote/weighted/judge) | `conflict_resolution.py` |
| L12 | Multi-agent 成本 (token 爆炸) | `cost_analyzer.py` |
| L13 | **Capstone**: 3-agent coding crew | `capstone_coding_crew.py` |

## Tags

- `multi-agent` — Module 7 第 4 专题

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/multi-agent-orchestration/src/tests/test_multi_agent.py
```

## 跑 Capstone

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/multi-agent-orchestration/src'); from capstone_coding_crew import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 6 框架选型决策树

| 场景 | 推荐 |
|------|------|
| 快速 PoC | CrewAI (role-based, 简洁) |
| 复杂状态机 | LangGraph |
| 研究/实验 | AutoGen |
| SOP 化软件开发 | MetaGPT |
| 通用 supervisor | Magentic-One |
| 极简 hand-off | Swarm |

## 关键文献

- AutoGen (Microsoft 2023.10)
- CrewAI (joaomdmoura 2024)
- LangGraph (LangChain 2024)
- MetaGPT (Hong 2023, DeepWisdom)
- Magentic-One (Microsoft 2024.11)
- OpenAI Swarm (2024.10, experimental)

## 一句话

> 6 框架共一个内核：N agent + 消息总线 + 终止条件 — 手写一遍看清 framework 是什么。
