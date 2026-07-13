# 14. ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition

- 年份/来源: 2025 / arXiv
- 方向分类: benchmark
- 本地论文: [14-researchbench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/14-researchbench.pdf)
- 在线来源: https://arxiv.org/abs/2503.21248
- 下载入口: https://arxiv.org/pdf/2503.21248

## 一句话定位
把科学发现拆成 inspiration-based 子任务，评估 LLM 如何从启发文献组合出发现。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学 idea 很少凭空出现，常来自跨论文启发和组合。ResearchBench 关注 LLM 是否能找到 inspiration、理解关系并形成 hypothesis。

## 方法拆解
benchmark 将 discovery 分解为启发检索、启发理解、假设组合和质量排序等步骤，从而定位模型在哪一步失败。

## 证据与实验怎么看
论文评估 LLM 在 inspiration-based discovery 上的表现，揭示当前模型在复杂组合和排序方面仍有限。

## 局限、风险和批判点
任务分解可能简化真实发现。真实 inspiration 不一定来自显式相关论文，也可能来自异常结果或跨领域类比。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
适合用来设计文献调研框架：不是只列 paper，而是问每篇给了什么 inspiration，能和哪些 paper 组合。

## 组会讨论问题
- 科学 discovery 中 inspiration 应如何标注？
- LLM 是不会找启发，还是不会判断启发价值？
- benchmark 能否捕捉跨领域迁移？

## 可复现或可延伸的 follow-up
对 30 篇做 inspiration graph：节点是论文，边是可组合出新方向的关系。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
