# 24. Aviary: training language agents on challenging scientific tasks

- 年份/来源: 2024 / arXiv
- 方向分类: agent_training
- 本地论文: [24-aviary.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/24-aviary.pdf)
- 在线来源: https://arxiv.org/abs/2412.21154
- 下载入口: https://arxiv.org/pdf/2412.21154

## 一句话定位
提供训练语言 agent 的科学任务环境，把 agent 能力从 prompt engineering 推向环境训练。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
很多 agent 系统只是手写 workflow。Aviary 关心如何像训练智能体一样训练 language agents，让它们在科学任务中通过交互学习。

## 方法拆解
它构建 challenging scientific tasks 和 agent environment，支持工具调用、状态反馈和策略学习。重点是训练环境，而不是单个应用 demo。

## 证据与实验怎么看
论文展示在科学任务上训练 agent 可以提升表现，并为后续研究提供 gym-like 平台。

## 局限、风险和批判点
环境设计会强烈塑造 agent 行为。若任务与真实科学差异大，训练出的策略迁移有限。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是 AI Scientist 的训练层基础设施：未来不只是 prompt 一个 scientist，而是训练一个能适应科研环境的 agent。

## 组会讨论问题
- 科研环境应暴露哪些 observation 和 action？
- reward 如何避免短视？
- 能否训练 agent 学会失败后改实验？

## 可复现或可延伸的 follow-up
为 video diffusion 做 mini Aviary：action 是改代码、跑实验、读结果，reward 是指标提升和报告质量。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
