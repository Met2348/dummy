# 30. AlphaEvolve: A coding agent for scientific and algorithmic discovery

- 年份/来源: 2025 / arXiv
- 方向分类: verifiable_discovery
- 本地论文: [30-alphaevolve.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/30-alphaevolve.pdf)
- 在线来源: https://arxiv.org/abs/2506.13131
- 下载入口: https://arxiv.org/pdf/2506.13131

## 一句话定位
将 coding agent、进化搜索和自动评估结合，用于算法和科学发现。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
FunSearch 证明程序搜索有效，AlphaEvolve 进一步扩展到更通用、更工程化的 coding agent discovery。

## 方法拆解
系统让 coding agent 生成和修改程序，使用自动 evaluator 评分，并通过进化过程保留和组合高质量候选。强调规模化搜索、工具执行和严格验证。

## 证据与实验怎么看
论文展示在算法优化和科学计算相关任务中取得新发现或改进结果。强点是每个结果都有可执行证据。

## 局限、风险和批判点
同样依赖可形式化问题和 evaluator。对开放式科学问题，如何把重要性写成评分函数仍困难。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
AlphaEvolve 是 AI Scientist 最值得借鉴的可靠路线：从自动写论文转向自动产生可验证 artifact。

## 组会讨论问题
- AI Scientist 是否应优先研究可验证 artifact？
- evolutionary search 与 tree search 如何取舍？
- 怎样把失败候选变成可复用知识？

## 可复现或可延伸的 follow-up
在 video-generation 方向，先让 agent 进化 temporal coherence metric、loss 或 schedule，产出可复现实验 artifact。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
