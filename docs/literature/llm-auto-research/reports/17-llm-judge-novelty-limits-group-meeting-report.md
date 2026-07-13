# 17. On the Limits of LLM-as-Judge for Scientific Novelty Assessment

- 年份/来源: 2026 / arXiv
- 方向分类: risk
- 本地论文: [17-llm-judge-novelty-limits.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/17-llm-judge-novelty-limits.pdf)
- 在线来源: https://arxiv.org/abs/2606.12071
- 下载入口: https://arxiv.org/pdf/2606.12071

## 一句话定位
2026 新近工作，专门质疑用 LLM-as-Judge 评估科学新颖性的可靠性。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
许多 AI Scientist pipeline 依赖 LLM judge 给 novelty、quality、acceptance 打分。如果 judge 不可靠，整个自动优化会被错误信号牵引。

## 方法拆解
论文分析 LLM judge 在 scientific novelty assessment 上的边界，比较不同 judge、提示、领域和人类评价的一致性。

## 证据与实验怎么看
核心结论是 LLM judge 对 novelty 的判断存在系统偏差，容易被表述、领域熟悉度和检索上下文影响。

## 局限、风险和批判点
结论依赖评测集和人类标注质量。但即便如此，它对自动科研系统是重要警告：评分器不是中立裁判。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
与 All That Glitters 一起构成风险侧核心阅读：一个讲 plagiarism，一个讲 judge failure。

## 组会讨论问题
- 如果 LLM judge 不可靠，AI Scientist 的搜索目标如何定义？
- novelty 是否必须结合检索和人工专家？
- 多 judge ensemble 能否降低偏差？

## 可复现或可延伸的 follow-up
建立双层 novelty check：先检索相似工作，再对差异进行结构化审核。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
