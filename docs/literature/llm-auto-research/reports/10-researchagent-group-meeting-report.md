# 10. ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models

- 年份/来源: 2024 / arXiv
- 方向分类: ideation
- 本地论文: [10-researchagent.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/10-researchagent.pdf)
- 在线来源: https://arxiv.org/abs/2404.07738
- 下载入口: https://arxiv.org/pdf/2404.07738

## 一句话定位
基于文献迭代生成 research idea，是从 literature graph 到 idea generation 的典型工作。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
自动科研第一步常常是读文献找 gap。ResearchAgent 不直接做实验，而是围绕文献检索、相关工作理解、idea 生成和迭代改进。

## 方法拆解
系统从用户研究兴趣出发检索相关论文，构建上下文，生成 idea，再通过进一步文献反馈迭代。核心是让 idea 扎根文献证据。

## 证据与实验怎么看
论文通过人工评价 idea 的 novelty、feasibility、relevance 展示提升。价值在于把 literature review 和 ideation 合并。

## 局限、风险和批判点
idea 评价主观性强，系统可能偏向热门文献附近的安全想法。检索库不全或排序偏置会锁死 idea 空间。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是老师任务最直接相关的工具型论文：如何从 30 篇文献提炼方向，而不是逐篇读完就结束。

## 组会讨论问题
- 文献驱动 idea generation 如何避免 incremental combination？
- 检索召回和 idea diversity 如何平衡？
- 一个好 idea card 应包含哪些字段？

## 可复现或可延伸的 follow-up
为本次 30 篇文献生成 idea cards：来源论文、未解决问题、最小实验、风险和可能投稿 venue。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
