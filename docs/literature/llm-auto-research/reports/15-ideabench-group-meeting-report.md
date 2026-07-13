# 15. IdeaBench: Benchmarking Large Language Models for Research Idea Generation

- 年份/来源: 2024 / arXiv
- 方向分类: benchmark
- 本地论文: [15-ideabench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/15-ideabench.pdf)
- 在线来源: https://arxiv.org/abs/2411.02429
- 下载入口: https://arxiv.org/pdf/2411.02429

## 一句话定位
面向 research idea generation 的基准，用于比较 LLM idea 质量。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
idea generation 太容易停留在 anecdote。IdeaBench 将 idea 质量拆成可评估维度，让模型和方法能比较。

## 方法拆解
它构建研究场景和评价标准，衡量新颖性、可行性、相关性、潜在影响等，通常结合自动评分与人工评分。

## 证据与实验怎么看
论文显示不同 LLM 在 idea generation 上差异明显，也提示 prompt、检索增强和迭代反馈会影响 idea 质量。

## 局限、风险和批判点
任何 idea benchmark 都面临 ground truth 缺失。评审者偏好、时间尺度和领域熟悉度都会影响评分。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
可作为如何评价 idea 的方法论补充，适合与大规模人类实验那篇搭配读。

## 组会讨论问题
- research idea 的好坏能否脱离执行结果评估？
- impact 这种长期变量如何短期打分？
- benchmark 是否奖励热门方向？

## 可复现或可延伸的 follow-up
为 PhD 方向候选做 IdeaBench 表：novelty、feasibility、advisor fit、compute cost 打分。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
