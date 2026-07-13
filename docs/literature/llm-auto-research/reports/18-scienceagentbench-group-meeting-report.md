# 18. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery

- 年份/来源: 2024 / arXiv / ICLR 2025
- 方向分类: benchmark
- 本地论文: [18-scienceagentbench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/18-scienceagentbench.pdf)
- 在线来源: https://arxiv.org/abs/2410.05080
- 下载入口: https://arxiv.org/pdf/2410.05080

## 一句话定位
给语言 agent 做 data-driven scientific discovery 的严格评测。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
很多 agent demo 看起来能做科学任务，但缺少可复现 benchmark。ScienceAgentBench 让系统在真实数据、代码和科学问题上接受考验。

## 方法拆解
benchmark 设计数据驱动科学任务，要求 agent 理解问题、操作数据、写代码、分析结果并提交答案。评价强调过程和最终科学结论。

## 证据与实验怎么看
论文比较不同 agent 的成功率，显示当前系统在长链条推理、数据处理和错误恢复上仍不稳定。

## 局限、风险和批判点
benchmark 覆盖有限，且评分标准会影响 agent 行为。真实科学发现的长期价值难以通过短任务完全体现。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
它提醒我们：会写论文不等于会处理真实科学数据。

## 组会讨论问题
- 科学 agent benchmark 应重代码执行，还是重科学解释？
- 失败更多来自工具使用、统计理解还是计划能力？
- 能否设计 video/world-model 版本？

## 可复现或可延伸的 follow-up
构建 5 个 video-generation data tasks：让 agent 诊断 temporal artifacts 并提出实验。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
