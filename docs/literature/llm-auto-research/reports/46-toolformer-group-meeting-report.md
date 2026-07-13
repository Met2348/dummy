# 46. Toolformer: Language Models Can Teach Themselves to Use Tools

- 年份/来源: 2023 / arXiv
- 作者: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer et al.
- 方向分类: tool_use (工具使用学习)
- 本地论文: [46-toolformer.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/46-toolformer.pdf)
- 在线来源: https://arxiv.org/abs/2302.04761
- 下载入口: https://arxiv.org/pdf/2302.04761

## 一句话定位
这篇关注模型如何学会调用外部工具，是科研 agent 可靠性的基础能力之一。

## 30 分钟组会讲法
- 0-3 min: 说明这篇论文处在 AI 自动科研链条的哪一层: literature、search、agent、tool、code、benchmark 或 discovery。
- 3-8 min: 讲背景和核心问题，说明为什么原有 LLM/RAG/agent 方法不够。
- 8-16 min: 拆系统或 benchmark 设计，画出模块、输入输出、评分器或反馈回路。
- 16-23 min: 讲实验和证据，尤其区分“任务完成”与“科研价值”。
- 23-27 min: 讲局限、失败模式、与已有 01-30 篇的关系。
- 27-30 min: 抛出讨论问题，落到本课题组可做的 follow-up。

## 背景和核心问题
科学任务需要计算器、检索器、数据库、代码执行器和绘图工具。模型若不会在合适时机调用工具，就会用语言猜答案。

## 摘要级内容抓手
根据摘要，这篇工作的核心关注点可以概括为：Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools...

这一段不是替代精读，而是帮组会开场快速定位：标题和摘要显示，这篇工作应被放在“工具使用学习”这一层，而不是泛泛当作又一个 LLM 应用。

## 方法拆解
重点讲工具调用数据如何构造、模型如何决定何时调用、如何把工具结果接回生成过程。

组会时建议画一张三层图：
1. 输入层: 用户问题、论文库、网页、代码库、数据集或交互环境。
2. 决策层: planner、retriever、reasoner、tool caller、critic、evaluator 或 memory。
3. 输出层: 报告、代码 patch、实验结果、benchmark 分数、hypothesis 或可验证 artifact。

## 证据与实验怎么看
看工具增强在数学、检索、问答等任务上是否减少错误，以及工具调用是否可控。

读实验部分时不要只摘最高分。要记录作者证明了什么、没有证明什么，以及这个证明是否能支撑“AI 自动科研”这个更大的 claim。

## 局限、风险和批判点
学会调用工具不等于会判断工具输出。工具错误、参数错误和过度调用仍是风险。

对老师汇报时可以主动讲一个反例：如果把这篇方法直接用于真实科研，最可能在哪一步失败，是检索、评价、工具调用、实验设计，还是 novelty 判断？

## 放在 70 篇图谱中的位置
这篇补的是 工具使用学习 这一块能力。它不一定直接产生科学发现，但它支撑 AI Scientist 的某个必要环节。

与已有 01-30 篇的关系: 01-09 更偏端到端 AI Scientist，10-17 偏文献/idea/novelty，18-24 偏科学 agent benchmark，25-30 偏领域发现和可验证发现。本篇应作为这些主线的补充能力或评测依据。

## 组会讨论问题
- 这篇论文解决的是 AI 自动科研链条中的哪一环，能否独立构成 PhD 方向？
- 它的评价指标是否真的衡量了科研质量，还是只衡量任务完成度？
- 如果迁移到 video generation / world models，应替换哪些工具、数据和 verifier？

## 可复现或可延伸的 follow-up
为科研 agent 定义最小工具集：paper search、PDF reader、code runner、stats checker、plotter、novelty checker。

## 建议 slides
1. 标题 + 本文属于哪一层能力。
2. 为什么需要这篇: 它补齐 70 篇图谱中的哪块缺口。
3. 方法/benchmark 总图。
4. 关键实验或任务设计。
5. 和 01-30 中最相近论文的对比。
6. 局限和失败模式。
7. 对 video generation / world models / AI Scientist 方向的启发。
8. 讨论问题和下一步可做实验。
