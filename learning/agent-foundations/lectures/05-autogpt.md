# L05 · AutoGPT / BabyAGI（2023 玩具开源）

## 历史地位

2023.03-04，**AutoGPT** 和 **BabyAGI** 两个开源玩具点燃了 agent 第一波热潮。
- 概念：给 LLM 一个 high-level 目标，让它**自循环规划 + 执行**直到完成。
- 现实：大多数任务跑不通（成本高、循环卡死、tool 滥用）。
- 但：**范式定义**了后续 LangChain / LangGraph / CrewAI 走向。

## AutoGPT 架构（v0.4）

```
Goal: "Research electric cars and write summary"
       ↓
  1. Plan (LLM 列任务)
       ↓
  2. Execute (pick next task, ReAct + tools)
       ↓
  3. Reflect (memory + 新 task)
       ↓
  4. Loop until "task complete"
```

## BabyAGI（Yohei Nakajima, 100 行 Python）

```python
objective = "do X"
task_list = ["Step 1"]
while task_list:
    task = task_list.pop(0)
    result = llm(f"Do: {task}\nObjective: {objective}")
    new_tasks = llm(f"Generate next tasks based on {result}")
    task_list.extend(new_tasks)
    save_to_vector_db(task, result)  # memory
```

## 为什么"玩具"

| 问题 | 表现 |
|------|------|
| 死循环 | 不会判断 "objective 已达成" |
| Token 爆炸 | history 越积越多 |
| Tool 滥用 | 反复 web search 同一关键词 |
| 无 HITL | 出错也不停 |

## 后世改进

| 框架 | 改 AutoGPT 哪 |
|------|--------------|
| LangGraph | 显式 StateGraph 防卡死 |
| CrewAI | 多 agent 拆任务 |
| Magentic-One | Orchestrator + worker |
| Claude Computer Use | 真控制屏幕 |

## 现代视角的 AutoGPT

```
不要直接用 AutoGPT。但是：
  ✓ 学其"自循环规划"思想
  ✓ 看其失败模式，反过来设计防御
  ✓ 历史地位：催生整个 agent 生态
```

## 关键 takeaway

| 设计教训 | 后世 framework 做法 |
|---------|---------------------|
| 必须有终止条件 | LangGraph END node / max_iter |
| 必须有 HITL | LangGraph interrupt() |
| 必须 trace 可观察 | LangSmith / W&B Weave |
| 必须 budget cap | OpenAI usage cap |

## 关键文献 / 项目

- AutoGPT (Significant-Gravitas 2023.03)
- BabyAGI (Yohei Nakajima 2023.04)
- AgentGPT, GodMode (web 玩具版)

## 退出条件

- 能讲 AutoGPT 4 步 loop
- 能列 4 个失败模式
- 知道现代 framework 如何修复

## 一句话

> AutoGPT/BabyAGI = 2023 agent 热潮起点 —— 范式正确，工程不及格，但启发了之后所有 agent 框架。
