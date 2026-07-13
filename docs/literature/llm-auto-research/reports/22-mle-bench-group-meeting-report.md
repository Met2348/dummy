# 22. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

- 年份/来源: 2024 / arXiv
- 方向分类: benchmark
- 本地论文: [22-mle-bench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/22-mle-bench.pdf)
- 在线来源: https://arxiv.org/abs/2410.07095
- 下载入口: https://arxiv.org/pdf/2410.07095

## 一句话定位
用 Kaggle 竞赛评估机器学习工程 agent，是检验 agent 实战工程能力的重要基准。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
AI Scientist 需要大量 ML engineering 能力：读数据、建 pipeline、调模型、提交结果。MLE-bench 用真实竞赛环境测试这一点。

## 方法拆解
benchmark 收集 Kaggle 风格任务，提供数据和评分方式，让 agent 在限制时间内改进方案，评价 leaderboard 分数和任务完成情况。

## 证据与实验怎么看
论文显示即使强模型在真实 ML engineering 上也困难，尤其是环境配置、数据处理、错误诊断和策略迭代。

## 局限、风险和批判点
Kaggle 成功不等于科学发现；它偏工程优化，可能奖励 stacking 和调参，而不是解释性贡献。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
MLE-bench 是 AI Scientist 的手脚能力测试。没有这类能力，系统很难自主完成实验。

## 组会讨论问题
- ML engineering benchmark 与 scientific research benchmark 的边界在哪里？
- agent 在 Kaggle 中失败最多的环节是什么？
- 是否应把资源成本纳入评分？

## 可复现或可延伸的 follow-up
给研究 agent 设置工程门槛：任何 proposed method 必须能在 clean environment 一键复现实验。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
