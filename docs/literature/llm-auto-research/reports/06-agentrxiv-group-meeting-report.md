# 06. AgentRxiv: Towards Collaborative Autonomous Research

- 年份/来源: 2025 / arXiv
- 方向分类: end_to_end
- 本地论文: [06-agentrxiv.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/06-agentrxiv.pdf)
- 在线来源: https://arxiv.org/abs/2503.18102
- 下载入口: https://arxiv.org/pdf/2503.18102

## 一句话定位
把自动科研从单个 agent lab 扩展到多个实验室共享 preprint 的协作生态。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学进步不是单个系统独立完成，而是许多研究者持续共享结果。AgentRxiv 问多个 autonomous labs 能否像人类一样上传报告、阅读彼此结果并迭代改进。

## 方法拆解
系统提供共享预印本服务器。多个 agent laboratory 围绕同一目标工作，检索先前报告，继承有用策略，避免重复错误，并上传自己的研究产物。

## 证据与实验怎么看
论文在 reasoning/prompting 技术开发任务上比较 isolated labs 与 collaborative labs，报告共享历史能带来性能提升。核心证据支持科研记忆和协作的重要性。

## 局限、风险和批判点
实验目标仍偏工程 benchmark。共享报告可能放大错误结论，若缺少审稿和可信度标注，agent 会互相污染。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
它引出一个关键方向：AI research agent 不只是做实验，还要维护可信研究记忆和社区级知识库。

## 组会讨论问题
- agent 如何区分可靠发现和噪声？
- 自动预印本需要怎样的 peer review 或 provenance 机制？
- 个人 PhD 仓库能否模拟一个单人 AgentRxiv？

## 可复现或可延伸的 follow-up
把 30 篇报告做成 personal AgentRxiv：每篇有 contribution、evidence、risk 和 follow-up。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
