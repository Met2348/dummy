# 16. All That Glitters is Not Novel: Plagiarism in AI Generated Research

- 年份/来源: 2025 / ACL 2025 / arXiv
- 方向分类: risk
- 本地论文: [16-all-that-glitters.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/16-all-that-glitters.pdf)
- 在线来源: https://arxiv.org/abs/2502.16487
- 下载入口: https://arxiv.org/pdf/2502.16487

## 一句话定位
ACL 2025 Outstanding Paper，提醒 AI 生成研究可能不是 novel，而是语义层面的 plagiarism。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
AI Scientist 最危险的幻觉不是事实错误，而是看起来新颖。本文研究 AI 生成研究内容是否在语义上复用了已有工作。

## 方法拆解
论文分析 AI 生成内容与已有文献的相似性，使用检测流程识别语义抄袭，把 novelty 从主观印象拉回文献证据。

## 证据与实验怎么看
作者展示 AI 生成内容中存在大量表面改写但核心贡献重复的情况，说明流畅度和新颖感不能证明真正创新。

## 局限、风险和批判点
语义相似检测可能误伤合理继承或独立发现。科学研究本来就建立在前人工作上，关键是区分合法延展与实质重复。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是所有 AI Scientist 论文都应配套阅读的风险约束。没有 novelty checking，自动科研会制造低质量 paper-like artifacts。

## 组会讨论问题
- 怎样定义语义抄袭：方法同、实验同还是 claim 同？
- novelty check 应在 idea 阶段还是论文阶段做？
- LLM 生成内容是否需要 provenance graph？

## 可复现或可延伸的 follow-up
给每个 proposed idea 做 novelty audit：列出最相近三篇论文和差异点。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
