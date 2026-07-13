# 07. Towards an AI co-scientist

- 年份/来源: 2025 / arXiv
- 方向分类: hypothesis
- 本地论文: [07-ai-co-scientist.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/07-ai-co-scientist.pdf)
- 在线来源: https://arxiv.org/abs/2502.18864
- 下载入口: https://arxiv.org/pdf/2502.18864

## 一句话定位
Google 多智能体 AI co-scientist，聚焦假设生成、辩论、演化和科学家协作。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
真实科研早期瓶颈常是假设空间太大，而不是论文写作。论文关注如何生成高质量、可检验、可排序的科学假设。

## 方法拆解
系统由 generation、reflection、ranking、evolution、meta-review 等 agent 构成。多个 agent 提出假设、互相批判、改写、排序，并围绕人类目标逐步收敛。

## 证据与实验怎么看
论文展示在生物医学等场景中的假设生成案例，并强调部分假设可与专家或实验验证相互印证。价值在于把科研创造力拆成多角色争论和演化。

## 局限、风险和批判点
假设生成容易显得惊艳，但真正价值取决于验证。系统可能产生看似合理但不可操作的假设，也可能被文献偏见限制。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
代表 co-scientist 路线：不必直接替代科学家，而是做假设生成和批判伙伴。PhD 阶段更现实的是这种 workflow。

## 组会讨论问题
- 多 agent debate 是否真的提升科学性？
- 假设排序应优先 novelty、feasibility 还是 impact？
- 如何让 co-scientist 输出直接变成实验 proposal？

## 可复现或可延伸的 follow-up
做方向选择 co-scientist：video generation、embodied AI、systems 三个 agent 互相批判，输出 4 年路线。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
