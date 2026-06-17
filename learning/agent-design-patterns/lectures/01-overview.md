# L01 · Agent Design 是什么 / 为什么需要"设计层"

## 30 秒定位

Module 7 前 7 个专题教你 agent 怎么**实现**(ReAct/Reflexion/StateGraph)和用哪个**框架**(LangGraph/CrewAI/Claude SDK)。本专题往上挪一层,回答两个更早的问题:

> **该不该造 agent?造成什么形状?**

这层叫 **agent design**——在写任何 `while` 循环前的架构决策。蓝本是 Anthropic《Building Effective Agents》(2024.12)。

## 一句话区分:Workflow vs Agent

| | Workflow(工作流) | Agent(智能体) |
|---|------------------|---------------|
| 控制流 | **代码预先写死**的路径 | **LLM 自己决定**下一步 |
| 可预测性 | 高(图是固定的) | 低(轨迹是涌现的) |
| 成本/延迟 | 可估算 | 不确定,可能爆 |
| 适用 | 任务能拆成已知步骤 | 步骤数/顺序事先不知道 |
| 例子 | "翻译→润色→校验" | "修好这个 bug"(改哪些文件未知) |

```
Workflow:  input → [step A] → [step B] → [step C] → output   (你画的图)
Agent:     input → LLM ⇄ tools (循环,直到模型说 done)        (模型走的路)
```

**核心命题(贯穿全专题)**:**模型固定时,架构本身就是性能/成本杠杆**。同一个 model,选 workflow 还是 agent,成本能差好几倍——capstone(L14)用一张表证明这点。

## 5 大 workflow 模式 + 1 个 agent

本专题的骨架(后续每节一个):

| 模式 | 一句话 | src |
|------|--------|-----|
| Prompt Chaining | 固定顺序拆步,步间加 gate | [prompt_chaining.py](../src/patterns/prompt_chaining.py) |
| Routing | 先分类,再分流到专门 handler | [routing.py](../src/patterns/routing.py) |
| Parallelization | 独立子任务并行 / 多次投票 | [parallelization.py](../src/patterns/parallelization.py) |
| Orchestrator-Workers | 动态拆子任务再派发 | [orchestrator_workers.py](../src/patterns/orchestrator_workers.py) |
| Evaluator-Optimizer | 生成-评估循环 | [evaluator_optimizer.py](../src/patterns/evaluator_optimizer.py) |
| **Autonomous Agent** | 开放循环,模型自己选工具 | [autonomous_agent.py](../src/patterns/autonomous_agent.py) |

外加三节"横切"内容:context engineering(L10)、agentic patterns + 12-Factor(L11)、反模式(L12)、设计 checklist(L13)。

## 这专题和 Module 7 其它专题的边界

- `agent-foundations` = **单 agent 循环原理**(机制层)
- `multi-agent-orchestration` = **多 agent 协作**(机制层)
- **本专题 = 横切两者的选型与权衡**(设计层):什么时候根本不该上 agent。

## 退出条件
- [ ] 能用一句话区分 workflow 和 agent
- [ ] 记住 5+1 模式的名字和各自适用场景
- [ ] 理解"架构是成本杠杆"这个核心命题
