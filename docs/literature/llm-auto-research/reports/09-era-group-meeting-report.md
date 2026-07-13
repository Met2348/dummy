# 09. An AI system to help scientists write expert-level empirical software

- 年份/来源: 2025 / arXiv
- 方向分类: experimentation
- 本地论文: [09-era.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/09-era.pdf)
- 在线来源: https://arxiv.org/abs/2509.06503
- 下载入口: https://arxiv.org/pdf/2509.06503

## 一句话定位
ERA 聚焦科学发现中的经验软件瓶颈，用 LLM + tree search 写专家级 empirical software。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
很多科研不是卡在 idea，而是卡在把实验软件写对、写强、写到专家水平。ERA 将经验科学软件生成看作可优化问题。

## 方法拆解
系统用 LLM 生成或修改程序，用 tree search 在候选程序空间探索。每个候选通过质量 metric 评估，搜索不断整合外部来源中的复杂研究想法。

## 证据与实验怎么看
论文展示 ERA 在不同科学和计算任务上达到专家级结果。强点是有明确 metric，因此比开放式论文生成更可验证。

## 局限、风险和批判点
需要可自动评估的质量指标；没有 metric 的科学问题难直接套用。tree search 也可能过拟合 benchmark 或产出难解释工程拼接。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
对 EE/工程背景很有价值。AI Scientist 的可靠切口之一不是自动写论文，而是自动写、测、改实验软件。

## 组会讨论问题
- 科研软件质量 metric 应如何定义？
- tree search 生成的软件如何做安全审查？
- video generation 中哪些实验指标适合 ERA 化？

## 可复现或可延伸的 follow-up
把 temporal coherence toy benchmark 做成 ERA 环境：agent 改 denoiser、schedule、loss，metric 是 coherence 与质量综合分。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
