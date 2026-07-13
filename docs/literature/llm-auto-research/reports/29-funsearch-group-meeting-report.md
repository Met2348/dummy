# 29. Mathematical discoveries from program search with large language models

- 年份/来源: 2024 / Nature
- 方向分类: verifiable_discovery
- 本地论文: [29-funsearch.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/29-funsearch.pdf)
- 在线来源: https://www.nature.com/articles/s41586-023-06924-6
- 下载入口: https://www.nature.com/articles/s41586-023-06924-6.pdf

## 一句话定位
LLM + 程序搜索做可验证数学和算法发现的 Nature 代表作。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
科学发现最可靠形式之一是可执行、可验证的程序。FunSearch 选择数学和算法问题，用 LLM 生成程序，用 evaluator 严格评分。

## 方法拆解
系统让 LLM 生成候选程序，将高分程序放入数据库，再用进化式循环不断改进。关键是评价函数确定且自动化。

## 证据与实验怎么看
论文展示在 cap set 等问题和 bin-packing 启发式上发现新结果或更好算法，证明 LLM 可通过搜索产生可验证发现。

## 局限、风险和批判点
适用范围依赖可执行表示和自动评分。很多科学假设没有即时 evaluator，难以直接套用。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是 AI Scientist 最重要的可验证发现范式之一：不要问模型想法是否好，让程序和 evaluator 说话。

## 组会讨论问题
- 哪些科研问题可以转成 program search？
- evaluator 设计会不会限制发现类型？
- 发现的程序如何解释成科学知识？

## 可复现或可延伸的 follow-up
把 loss、schedule、metric 设计转成 FunSearch：程序生成候选方法，metric 自动评估。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
