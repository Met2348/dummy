# 37. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents

- 年份/来源: 2025 / arXiv
- 作者: Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao
- 方向分类: deep_research_benchmark (Deep Research agent 评测)
- 本地论文: [37-deepresearch-bench.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/37-deepresearch-bench.pdf)
- 在线来源: https://arxiv.org/abs/2506.11763
- 下载入口: https://arxiv.org/pdf/2506.11763

## 一句话定位
这篇属于 Deep Research benchmark，用来衡量 agent 能否做长链条检索、综合和报告。

## 30 分钟组会讲法
- 0-3 min: 说明这篇论文处在 AI 自动科研链条的哪一层: literature、search、agent、tool、code、benchmark 或 discovery。
- 3-8 min: 讲背景和核心问题，说明为什么原有 LLM/RAG/agent 方法不够。
- 8-16 min: 拆系统或 benchmark 设计，画出模块、输入输出、评分器或反馈回路。
- 16-23 min: 讲实验和证据，尤其区分“任务完成”与“科研价值”。
- 23-27 min: 讲局限、失败模式、与已有 01-30 篇的关系。
- 27-30 min: 抛出讨论问题，落到本课题组可做的 follow-up。

## 背景和核心问题
浏览器 agent 或 RAG agent 的短问答评测不够，老师要求的深度调研正是长任务：多源检索、交叉验证、写报告、处理冲突证据。Deep Research benchmark 试图评测这种能力。

## 摘要级内容抓手
根据摘要，这篇工作的核心关注点可以概括为：Deep Research Agents are a prominent category of LLM-based agents. By autonomously orchestrating multistep web exploration, targeted retrieval, and higher-order synthesis, they transform vast amounts of online information into analyst-grade, citation-rich reports--compressing hours of manual desk research into minutes. However, a comprehensive benchmark for...

这一段不是替代精读，而是帮组会开场快速定位：标题和摘要显示，这篇工作应被放在“Deep Research agent 评测”这一层，而不是泛泛当作又一个 LLM 应用。

## 方法拆解
重点看 benchmark 如何设计任务、评分 rubrics、专家报告、证据引用和错误类型。一个好 benchmark 应能区分浅层搜索、拼接式综述和真正综合。

组会时建议画一张三层图：
1. 输入层: 用户问题、论文库、网页、代码库、数据集或交互环境。
2. 决策层: planner、retriever、reasoner、tool caller、critic、evaluator 或 memory。
3. 输出层: 报告、代码 patch、实验结果、benchmark 分数、hypothesis 或可验证 artifact。

## 证据与实验怎么看
组会时讲模型/agent 在 benchmark 上的失败模式：漏检关键来源、引用不支持结论、报告结构混乱、无法发现冲突。

读实验部分时不要只摘最高分。要记录作者证明了什么、没有证明什么，以及这个证明是否能支撑“AI 自动科研”这个更大的 claim。

## 局限、风险和批判点
Deep Research 评测很依赖 rubrics 和参考答案。若任务本身过窄，会奖励检索技巧；若太开放，又难评分。

对老师汇报时可以主动讲一个反例：如果把这篇方法直接用于真实科研，最可能在哪一步失败，是检索、评价、工具调用、实验设计，还是 novelty 判断？

## 放在 70 篇图谱中的位置
这篇补的是 Deep Research agent 评测 这一块能力。它不一定直接产生科学发现，但它支撑 AI Scientist 的某个必要环节。

与已有 01-30 篇的关系: 01-09 更偏端到端 AI Scientist，10-17 偏文献/idea/novelty，18-24 偏科学 agent benchmark，25-30 偏领域发现和可验证发现。本篇应作为这些主线的补充能力或评测依据。

## 组会讨论问题
- 这篇论文解决的是 AI 自动科研链条中的哪一环，能否独立构成 PhD 方向？
- 它的评价指标是否真的衡量了科研质量，还是只衡量任务完成度？
- 如果迁移到 video generation / world models，应替换哪些工具、数据和 verifier？

## 可复现或可延伸的 follow-up
把导师给的 70 篇任务本身变成一个 mini DeepResearch benchmark：评分覆盖率、去重、引用、报告深度和可追问性。

## 建议 slides
1. 标题 + 本文属于哪一层能力。
2. 为什么需要这篇: 它补齐 70 篇图谱中的哪块缺口。
3. 方法/benchmark 总图。
4. 关键实验或任务设计。
5. 和 01-30 中最相近论文的对比。
6. 局限和失败模式。
7. 对 video generation / world models / AI Scientist 方向的启发。
8. 讨论问题和下一步可做实验。
