# 42. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent

- 年份/来源: 2025 / arXiv
- 作者: Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu et al.
- 方向分类: browsing_benchmark (浏览/检索 agent 评测)
- 本地论文: [42-browsecomp-plus.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/42-browsecomp-plus.pdf)
- 在线来源: https://arxiv.org/abs/2508.06600
- 下载入口: https://arxiv.org/pdf/2508.06600

## 一句话定位
这篇衡量 browsing agent 在开放网络中找证据的能力，是深度文献调研的检索层基础。

## 30 分钟组会讲法
- 0-3 min: 说明这篇论文处在 AI 自动科研链条的哪一层: literature、search、agent、tool、code、benchmark 或 discovery。
- 3-8 min: 讲背景和核心问题，说明为什么原有 LLM/RAG/agent 方法不够。
- 8-16 min: 拆系统或 benchmark 设计，画出模块、输入输出、评分器或反馈回路。
- 16-23 min: 讲实验和证据，尤其区分“任务完成”与“科研价值”。
- 23-27 min: 讲局限、失败模式、与已有 01-30 篇的关系。
- 27-30 min: 抛出讨论问题，落到本课题组可做的 follow-up。

## 背景和核心问题
自动科研不能只靠模型内知识，必须能主动搜索、打开网页、比较来源、处理动态信息。浏览 benchmark 评估的就是这一层。

## 摘要级内容抓手
根据摘要，这篇工作的核心关注点可以概括为：Deep-Research agents, which integrate large language models (LLMs) with search tools, have shown success in improving the effectiveness of handling complex queries that require iterative search planning and reasoning over search results. Evaluations on current benchmarks like BrowseComp relies on black-box live web search APIs, have notable limitations in...

这一段不是替代精读，而是帮组会开场快速定位：标题和摘要显示，这篇工作应被放在“浏览/检索 agent 评测”这一层，而不是泛泛当作又一个 LLM 应用。

## 方法拆解
方法上关注任务构造、答案验证、网页环境、是否防止数据泄漏，以及 agent 需要多少步才能找到可靠证据。

组会时建议画一张三层图：
1. 输入层: 用户问题、论文库、网页、代码库、数据集或交互环境。
2. 决策层: planner、retriever、reasoner、tool caller、critic、evaluator 或 memory。
3. 输出层: 报告、代码 patch、实验结果、benchmark 分数、hypothesis 或可验证 artifact。

## 证据与实验怎么看
读实验时要看成功率之外的过程指标：错误来源、无效浏览步数、是否引用权威来源、能否处理多个来源冲突。

读实验部分时不要只摘最高分。要记录作者证明了什么、没有证明什么，以及这个证明是否能支撑“AI 自动科研”这个更大的 claim。

## 局限、风险和批判点
浏览 benchmark 不等于科研 benchmark。会找网页不代表会判断论文贡献；它只是 literature pipeline 的前端能力。

对老师汇报时可以主动讲一个反例：如果把这篇方法直接用于真实科研，最可能在哪一步失败，是检索、评价、工具调用、实验设计，还是 novelty 判断？

## 放在 70 篇图谱中的位置
这篇补的是 浏览/检索 agent 评测 这一块能力。它不一定直接产生科学发现，但它支撑 AI Scientist 的某个必要环节。

与已有 01-30 篇的关系: 01-09 更偏端到端 AI Scientist，10-17 偏文献/idea/novelty，18-24 偏科学 agent benchmark，25-30 偏领域发现和可验证发现。本篇应作为这些主线的补充能力或评测依据。

## 组会讨论问题
- 这篇论文解决的是 AI 自动科研链条中的哪一环，能否独立构成 PhD 方向？
- 它的评价指标是否真的衡量了科研质量，还是只衡量任务完成度？
- 如果迁移到 video generation / world models，应替换哪些工具、数据和 verifier？

## 可复现或可延伸的 follow-up
把 arXiv、ACL、Nature、OpenReview 的下载和核验流程标准化，减少人工找 PDF 的时间。

## 建议 slides
1. 标题 + 本文属于哪一层能力。
2. 为什么需要这篇: 它补齐 70 篇图谱中的哪块缺口。
3. 方法/benchmark 总图。
4. 关键实验或任务设计。
5. 和 01-30 中最相近论文的对比。
6. 局限和失败模式。
7. 对 video generation / world models / AI Scientist 方向的启发。
8. 讨论问题和下一步可做实验。
