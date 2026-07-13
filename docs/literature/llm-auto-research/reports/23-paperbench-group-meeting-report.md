# 23. PaperBench: Evaluating AI's Ability to Replicate AI Research

- 年份/来源: 2025 / arXiv
- 方向分类: benchmark
- 本地论文: [23-paperbench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/23-paperbench.pdf)
- 在线来源: https://arxiv.org/abs/2504.01848
- 下载入口: https://arxiv.org/pdf/2504.01848

## 一句话定位
评估 AI 复现 AI 论文的能力，是自动科研非常关键但常被低估的一环。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学不是只提出 idea，还要能理解和复现已有 work。PaperBench 问 AI 能否从论文出发复现研究贡献。

## 方法拆解
benchmark 选取 AI 研究论文，要求 agent 根据论文内容实现和运行关键方法，比较结果与原论文目标，评价复现质量和代码正确性。

## 证据与实验怎么看
论文显示当前 AI 在完整复现研究上仍有明显困难。这说明自动写新论文之前，自动理解旧论文仍是硬瓶颈。

## 局限、风险和批判点
复现任务评分和 ground truth 需要仔细设计。不同论文复现难度差异大，环境和缺失细节会严重影响结果。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
对 PhD 训练也重要：能复现是提出新贡献的前提。AI Scientist 可从自动复现切入。

## 组会讨论问题
- 复现失败应归因于 agent 还是原论文不可复现？
- 如何处理未公开细节？
- 自动复现系统如何生成 failure report？

## 可复现或可延伸的 follow-up
选 3 篇 video generation 小论文做 PaperBench-style 复现任务。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
