# 05. DeepScientist: Advancing Frontier-Pushing Scientific Findings Progressively

- 年份/来源: 2025 / arXiv
- 方向分类: end_to_end
- 本地论文: [05-deep-scientist.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/05-deep-scientist.pdf)
- 在线来源: https://arxiv.org/abs/2509.26603
- 下载入口: https://arxiv.org/pdf/2509.26603

## 一句话定位
把 AI Scientist 推向长周期目标导向 discovery，强调 findings memory 和分层验证。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
早期系统多是短任务、短实验、短论文。DeepScientist 关注持续推进人类定义的科学目标，通过多轮假设、验证和分析逐步超过已有方法。

## 方法拆解
它将 discovery 形式化为类似 Bayesian optimization 的过程，用 hypothesize、verify、analyze 循环推进。Findings Memory 记录已验证和失败发现，分层验证让低成本候选逐步进入高成本实验。

## 证据与实验怎么看
论文报告消耗大量 GPU 小时，生成数千 idea、验证上千候选，并在若干 AI 任务上超过人类设计的 SOTA。它强调长期累积，而非一次性生成论文。

## 局限、风险和批判点
这是高资源路线，算力门槛很高。发现到底来自 agent 的科学推理，还是来自大规模搜索和已有知识组合，需要谨慎判断。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
对 PhD 很有启发：四年研究不是单篇论文，而是 findings memory 的积累。个人科研仓库也应像 findings memory，而不是文件堆。

## 组会讨论问题
- 长周期 AI Scientist 的核心资产是代码、结果还是 findings memory？
- 如何防止大规模搜索在 benchmark 上过拟合？
- 低成本验证到高成本验证的升级准则是什么？

## 可复现或可延伸的 follow-up
在仓库里建立 findings memory：每篇文献对应 gap，每个 gap 对应最小实验，每个实验有 outcome。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
