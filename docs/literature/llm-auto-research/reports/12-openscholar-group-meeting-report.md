# 12. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs

- 年份/来源: 2024 / arXiv
- 方向分类: literature
- 本地论文: [12-openscholar.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/12-openscholar.pdf)
- 在线来源: https://arxiv.org/abs/2411.14199
- 下载入口: https://arxiv.org/pdf/2411.14199

## 一句话定位
开放科学文献 RAG 系统，强调大规模 paper datastore 和可复现文献综合。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学助手常被封闭检索和不可复现数据源限制。OpenScholar 试图构建开放、可检索、覆盖大规模科学论文的系统。

## 方法拆解
它结合大规模文献库、检索器和语言模型，用 retrieval-augmented generation 生成带引用的综述式回答。重点是开放数据源和系统工程。

## 证据与实验怎么看
论文在科学问答、综述生成等任务中与其他系统比较，展示开放检索增强模型可以接近或超过闭源方案。

## 局限、风险和批判点
大规模开放库不自动保证引用质量。真正难的是检索到反例、负结果和领域内被忽视的关键工作。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
对应基础设施层：如何建领域文献库，而不是每次从搜索引擎重来。

## 组会讨论问题
- 开放文献库 coverage 如何评估？
- 综述生成是否应显式列出未覆盖区域？
- 新兴方向检索不足会怎样影响 idea？

## 可复现或可延伸的 follow-up
为 LLM Auto Research 建小型 OpenScholar：papers、notes、citation graph、claims、counterclaims 分开存。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
