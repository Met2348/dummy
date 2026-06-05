# L06 · Plan-and-Execute

## 30 秒核心

> Plan-and-Execute = **先一次性把所有步骤规划好**，然后顺序执行 → 与 ReAct 的"边走边想"对立。

## 范式对照

| 范式 | 决策时机 | 代表 |
|------|---------|------|
| ReAct | 每步实时决 | LangChain ReAct |
| Plan-Execute | 开局全规划 | LangChain plan_and_execute |
| Plan-Solve | plan + 自检 | Plan-and-Solve (Wang 2023) |
| ReWOO | plan + 并行执行 + observation 后置 | ReWOO (Xu 2023) |

## 流程

```
Question
  ↓
[Planner LLM]
  ↓
Plan:
  1. search "topic A"
  2. search "topic B"
  3. summarize both
  4. final answer
  ↓
[Executor] (顺序执行 1-4)
  ↓
Final Answer
```

## 优缺点

| 优 | 缺 |
|---|----|
| 步骤可见，易审查 | 规划错就全错 |
| 可并行执行 (ReWOO) | 缺乏中途调整 |
| Token 节省 (无 reasoning 循环) | 复杂任务难 plan |

## ReWOO 改进（Xu 2023）

```
Planner: 给所有 step 留 placeholder #E1, #E2 ...
         step1 用 search → result 存 #E1
         step2 用 calc(#E1) → #E2
         ...
Worker: 并行执行（保 dependency order）
Solver: 用所有 #E 合成 final
```

→ Token-efficient（plan/solver 调 LLM，worker 调 tool）

## 何时用 Plan-Execute

| 场景 | 推荐 | 原因 |
|------|------|------|
| 数据分析 pipeline | ✓ | 步骤可预先列 |
| Search + summarize | ✓ | 顺序固定 |
| 交互式 chat | ✗ | 用户中途变意图 |
| 推理任务 | ✗ | 不知道下一步 |

## 实现核心（`plan_execute.py` 预告）

```python
def plan_execute(question, llm, tools):
    plan = llm(f"Question:{question}\nPlan as numbered list:")
    steps = parse_plan(plan)
    obs = {}
    for i, step in enumerate(steps):
        action = parse_action(step, obs)
        obs[f"E{i}"] = tools[action](args)
    return llm(f"Plan:{plan}\nObs:{obs}\nFinal Answer:")
```

## 关键文献

- Plan-and-Solve Prompting (Wang 2023)
- ReWOO: Decoupling Reasoning from Observations (Xu 2023)

## 退出条件

- 能讲 ReAct vs Plan-Execute 决策时机差异
- 知道 ReWOO 用 placeholder 实现并行

## 一句话

> Plan-and-Execute = 先列 plan 再顺序执行 —— 与 ReAct 互补，适合步骤可预先列出的任务。
