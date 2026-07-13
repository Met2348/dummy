# 19. DiscoveryBench: Towards Data-Driven Discovery with Large Language Models

- 年份/来源: 2024 / arXiv / ICLR 2025
- 方向分类: benchmark
- 本地论文: [19-discoverybench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/19-discoverybench.pdf)
- 在线来源: https://arxiv.org/abs/2407.01725
- 下载入口: https://arxiv.org/pdf/2407.01725

## 一句话定位
面向 data-driven discovery，考察 LLM 是否能从数据中发现关系和规律。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学发现不只是读论文，还包括从数据中提出和验证规律。DiscoveryBench 将 discovery 任务具体化，让模型面对真实数据。

## 方法拆解
benchmark 提供数据集和问题，要求模型进行分析、假设生成、验证和结论表述。重点是发现过程，而不是单步问答。

## 证据与实验怎么看
论文评估多种 LLM/agent 方法，揭示当前模型在数据理解、统计严谨性和因果区分上容易犯错。

## 局限、风险和批判点
数据驱动发现 benchmark 往往有隐含答案，可能更像考试而不是真发现。模型也可能通过模式匹配得到正确结论。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
它补上 literature-only 系统缺失的一环：AI Scientist 必须能从实验数据中学习。

## 组会讨论问题
- 什么数据任务能代表发现而非分析作业？
- LLM 如何处理负结果和不显著结果？
- 自动发现系统是否需要统计检验约束？

## 可复现或可延伸的 follow-up
用 toy video diffusion 数据构造任务：让 agent 发现 joint 模型为何比 per-frame coherent。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
