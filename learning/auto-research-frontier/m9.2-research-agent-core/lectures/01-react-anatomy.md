# L01 · ReAct 四零件解剖

## 1. 一个研究 agent 的最小骨架

抛开宣传，几乎所有"研究 agent"都是同一个骨架的放大版（综述
[2505.13259](https://arxiv.org/abs/2505.13259)、多 agent 架构
[2502.18864](https://arxiv.org/abs/2502.18864)）：

```
规划(Planning) ─ 工具(Tool use) ─ 记忆(Memory) ─ 角色(Roles)
```

`agent.py` 的 `ResearchAgent.run()` 把这四样串成一个 ReAct 闭环：

```python
subqs    = llm.decompose(question)      # 规划：拆问题
for sq in subqs:                        # 工具：逐子问题检索
    hits = search(sq, k=2)
    mem.add("search", ...)              # 记忆：scratchpad 累积观察
draft    = llm.draft_idea(...)          # 角色 Researcher：起草
critique = critic.review(draft, ...)    # 角色 Reviewer：独立批判
plan     = llm.revise(draft, critique)  # 按批判修正
```

## 2. 这四样正好是你 M7 学过的四个模块

| 零件 | 你学过的模块 | 这里怎么用 |
|------|------------|-----------|
| 规划 | agent-foundations（ReAct） | `decompose` 把大问题拆成可检索的子问题 |
| 工具 | tool-use-mcp | `corpus.search` 把"检索"抽象成一个可调用工具 |
| 记忆 | agent-memory-context | `Scratchpad` 记下每步，供回看与最终汇总 |
| 角色 | multi-agent-orchestration | Researcher 生成 / Reviewer 批判，职责分离 |

> **9.2 不是新东西，是把你已经学过的四块积木，按"做研究"这个任务拼起来。**
> AI Scientist-v2 的"实验经理 agent + 树搜索"、co-scientist 的"四 agent 辩论进化"，
> 都是在这四块上加深度，不是另起炉灶。

## 3. 为什么是闭环，不是直线

最朴素的写法是直线：拆问题→检索→写计划，完事。但那样**没有回路去纠错**。
ReAct/Reflexion 的关键是在产出后面接一个"观察—批判—修正"的回路。本模块用最小形式体现它：
`draft → critique → revise`。下一讲你会看到，这个回路删掉与否，直接决定最终计划里
有没有一条幻觉引用。

## 4. 动手

1. 读 `run.py` 打印的 `[ReAct transcript]`，对着上面的代码，指认出"规划/工具/记忆/角色"各在哪一行。
2. 把 `ResearchAgent(k_per_subq=2)` 改成 `k_per_subq=1`，检索变窄，transcript 怎么变？
   接地的论文变少后，下一讲的 critic 会不会更容易/更难抓幻觉引用？
