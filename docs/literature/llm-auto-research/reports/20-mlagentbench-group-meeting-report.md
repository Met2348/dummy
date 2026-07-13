# 20. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation

- 年份/来源: 2023 / arXiv / ICML 2024
- 方向分类: benchmark
- 本地论文: [20-mlagentbench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/20-mlagentbench.pdf)
- 在线来源: https://arxiv.org/abs/2310.03302
- 下载入口: https://arxiv.org/pdf/2310.03302

## 一句话定位
早期评估语言 agent 做机器学习实验的代表 benchmark。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
AI research agent 最接近可评测的场景是 ML experimentation：读任务、改模型、跑实验、看指标。

## 方法拆解
它给 agent 一组机器学习任务和代码环境，要求 agent 通过实验提升性能。评估最终分数、执行成功、实验过程和资源使用。

## 证据与实验怎么看
论文显示语言 agent 可以完成部分 ML 实验，但在长期规划、debug 和避免无效尝试上仍弱。

## 局限、风险和批判点
任务多为已有 ML 问题，离原创研究有距离，也可能奖励工程调参而非科学理解。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是 AI Scientist 的底层能力测试：如果连基本 ML 实验都不稳定，就很难相信能自动科研。

## 组会讨论问题
- 瓶颈是 coding、planning 还是 interpretation？
- 最终指标提升能否代表 research contribution？
- 如何记录失败实验避免重复踩坑？

## 可复现或可延伸的 follow-up
把 learning modules 转成 MLAgentBench 风格任务，评估 agent 能否复现实验并提出 ablation。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
