# 9.2 · 研究 Agent 的内核（Research Agent Core）

> 9.1 教你**从外面**给系统分级；9.2 把盖子掀开，看一个研究 agent **内部由什么零件构成**，
> 并亲手跑通一个 ReAct 闭环：`问题 → 检索 → 起草 idea → 自我批判 → 结构化研究计划`。

## 四个零件（正好是 M7 的四个模块在这里合体）

| 零件 | 在本模块的体现 | 桥接 |
|------|---------------|------|
| **规划 Planning** | `decompose` 把研究问题拆成子问题再喂检索 | agent-foundations |
| **工具 Tool use** | `corpus.search` 是外部检索工具（mock arXiv） | tool-use-mcp |
| **记忆 Memory** | `Scratchpad` 累积每步观察（ReAct scratchpad） | agent-memory-context |
| **多角色 Roles** | 生成=Researcher(`MockLLM`)，批判=Reviewer(`critic`)，职责分离 | multi-agent-orchestration |

`llm` 可插拔：默认确定性 `MockLLM`；接真模型只需实现 `decompose/draft_idea/revise` 三个方法。

## 这个模块要让你亲眼看到的事

研究 agent 最危险的失败之一是**幻觉引用**——煞有介事地引一篇根本不存在的文献
（[Hidden Pitfalls 2509.08713](https://arxiv.org/abs/2509.08713) 实证过）。本模块的
`MockLLM` 在草稿里**故意混入一个检索集外的引用 id**，确定性地复现这个失败，然后：

```powershell
python src/run.py            # 并排对比 有/无 critic
```

输出的关键一行：

```
无 critic 残留幻觉引用: 1  →  有 critic 残留: 0
```

**同一个生成模型、同一份检索，差别只在是否有一个职责分离的 Reviewer 去查引用。**
规划+工具+记忆让 agent 能跑，但只有独立批判才让它**不自欺**。

## 跑起来

```powershell
python src/run.py                                   # 默认：A 无critic / B 有critic 并排
python src/run.py --question "tree search research agent"   # 换问题
python src/run.py --no-critic                       # 只看裸 ReAct（幻觉引用残留）

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.2-research-agent-core `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.2-research-agent-core/
├── runbook.yaml                  V0/V1/V2 入口
├── lectures/
│   ├── 01-react-anatomy.md            ReAct 四零件解剖
│   ├── 02-tool-grounding-vs-hallucination.md  接地 vs 幻觉引用（核心）
│   └── 03-reviewer-role-reflexion.md  Reviewer 角色：批判为何要职责分离
└── src/
    ├── run.py
    ├── research_agent/
    │   ├── corpus.py             mock arXiv 检索库 + 确定性 search()
    │   ├── agent.py              ReAct 闭环 + MockLLM + Scratchpad 记忆
    │   └── critic.py            Reviewer 角色：查幻觉引用/缺baseline/吹新颖度
    └── tests/test_agent.py       7 测试：确定性 + critic 真清掉幻觉引用
```

## Hands-on（轮到你）

1. **加一种批判**：`critic.review` 现在查"幻觉引用/缺 baseline/吹新颖度"。
   再加一条——比如"子问题没覆盖到检索结果"（检索回来的论文一篇都没被引）。
   重跑看它能不能抓出新问题。
2. **接真检索**：把 `corpus.search` 换成对 `papers/INDEX.md` 全 40 篇的关键词检索，
   看更大检索集下幻觉引用还好不好抓。
3. **接真 LLM**：实现一个 `OpenAILLM.decompose/draft_idea/revise`，传给 `ResearchAgent(llm=...)`，
   观察真模型会不会也幻觉引用（大概率会）——这时 critic 就从教学玩具变成真守卫。

## 桥接

- 往回：agent-foundations（ReAct）· tool-use-mcp · agent-memory-context · multi-agent-orchestration。
- 往前：**9.3** 把"起草 idea"这一步放大成"生成 K 个 idea 并排序"，并直面 novelty≠feasibility；
  **9.4** 把"检索接地"放大成"带引用综述 + 引用忠实度核查"；**9.8** 红队这些幻觉。
