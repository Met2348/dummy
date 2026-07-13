# 02. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search

- 年份/来源: 2025 / arXiv
- 方向分类: end_to_end
- 本地论文: [02-ai-scientist-v2.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/02-ai-scientist-v2.pdf)
- 在线来源: https://arxiv.org/abs/2504.08066
- 下载入口: https://arxiv.org/pdf/2504.08066

## 一句话定位
第二代 AI Scientist，用 agentic tree search 探索研究空间，目标推进到 workshop-level 论文。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
第一代流水线探索较浅，容易早早固定一个 idea。v2 把研究过程看作搜索问题：idea、实验设计、结果解释、论文修改都可以形成分支，通过评估选择更有希望的路径。

## 方法拆解
核心是 agentic tree search。每个节点对应一个研究状态，包括假设、代码、实验结果和论文草稿。系统不断生成、评估、扩展候选研究路径，再用评分器剪枝或继续探索。

## 证据与实验怎么看
论文报告更强的自动化研究产出，并用 workshop-style review 衡量生成论文质量。贡献在于把 AI research automation 从 pipeline thinking 推到 search thinking。

## 局限、风险和批判点
tree search 带来更高算力和评估成本。更关键的是评分器偏差：如果评分器偏好论文外观，系统就会优化包装而非贡献。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是理解 2025 年后自动科研系统的重要节点。很多系统本质上都在换搜索空间、评分器或执行环境。

## 组会讨论问题
- research tree 的节点应该存储什么才能累积发现？
- 评分器如何避免奖励包装而非真实贡献？
- tree search 是否适合自动探索 ablation matrix？

## 可复现或可延伸的 follow-up
把研究方向选择写成 tree search：根节点是大方向，子节点是 gap，叶节点是最小实验。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
