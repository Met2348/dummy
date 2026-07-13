# 26. ChemCrow: Augmenting large-language models with chemistry tools

- 年份/来源: 2024 / Nature Machine Intelligence
- 方向分类: domain_chemistry
- 本地论文: [26-chemcrow.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/26-chemcrow.pdf)
- 在线来源: https://www.nature.com/articles/s42256-024-00832-8
- 下载入口: https://www.nature.com/articles/s42256-024-00832-8.pdf

## 一句话定位
把 LLM 与 18 个化学工具连接，展示工具增强在科学 agent 中的必要性。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
纯 LLM 不能可靠计算分子性质、查数据库或规划合成。ChemCrow 主张科学 agent 应 tool-augmented。

## 方法拆解
系统集成多个专用化学工具，包括数据库查询、分子计算、合成相关工具等。LLM 负责规划调用顺序和解释结果。

## 证据与实验怎么看
论文展示工具增强后在多种化学任务上优于纯 LLM，并能完成更复杂的任务链。

## 局限、风险和批判点
工具越多，错误传播和工具选择问题越严重。LLM 可能误解工具输出，或调用不合适工具。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
一般启示是 scientist agent 必须有领域工具箱。文献检索、代码执行、统计检验、可视化都应工具化。

## 组会讨论问题
- 工具增强系统如何验证每一步输出？
- 工具选择是否应由 planner 单独学习？
- 通用科研工具箱和领域工具箱怎样分层？

## 可复现或可延伸的 follow-up
定义 AI/ML research agent 工具箱：arXiv、Semantic Scholar、GitHub、pytest、wandb、plotting。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
