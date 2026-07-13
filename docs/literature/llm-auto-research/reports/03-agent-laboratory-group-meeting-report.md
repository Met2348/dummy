# 03. Agent Laboratory: Using LLM Agents as Research Assistants

- 年份/来源: 2025 / arXiv
- 方向分类: end_to_end
- 本地论文: [03-agent-laboratory.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/03-agent-laboratory.pdf)
- 在线来源: https://arxiv.org/abs/2501.04227
- 下载入口: https://arxiv.org/pdf/2501.04227

## 一句话定位
把自动科研做成实验室助手，而不是完全替代人，强调 literature review、experiment、report 三阶段协作。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
当前系统难以独立完成高质量科研，但可以承担大量重复科研劳动。论文的问题是如何把 LLM agent 变成可插入科研流程的研究助理。

## 方法拆解
系统分为文献综述 agent、实验 agent 和写作 agent。文献模块收集相关工作，实验模块改代码和跑结果，写作模块整理报告或论文草稿，人类在关键节点反馈。

## 证据与实验怎么看
论文展示 agent 协助完成从调研到实验报告的流程。价值不在某个算法，而在于把研究工作流工程化，让每个阶段有清晰输入输出。

## 局限、风险和批判点
系统容易变成自动化脚手架，科学性依赖人类给定问题和验收标准。缺少严格 benchmark 时，很难证明提升的是研究质量而非写作速度。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
对组会调研最实用：你现在要做的 30 篇文献包，本质就是一个 human-in-the-loop agent laboratory 的人工版。

## 组会讨论问题
- 哪些科研步骤必须由人做最终判断？
- 如何记录 agent 的失败尝试，使其成为研究资产？
- 组会报告能否作为 agent laboratory 的标准中间产物？

## 可复现或可延伸的 follow-up
建立 mini agent-lab：输入 30 篇论文，输出方向图谱、gap 表、实验 proposal 和每周组会报告。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
