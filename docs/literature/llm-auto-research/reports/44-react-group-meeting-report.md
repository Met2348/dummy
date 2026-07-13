# 44. ReAct: Synergizing Reasoning and Acting in Language Models

- 年份/来源: 2022 / arXiv
- 作者: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan et al.
- 方向分类: agent_foundation (reasoning + acting 基础范式)
- 本地论文: [44-react.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/44-react.pdf)
- 在线来源: https://arxiv.org/abs/2210.03629
- 下载入口: https://arxiv.org/pdf/2210.03629

## 一句话定位
这篇是 LLM agent 的基础工作，定义了模型如何交替推理和行动。

## 30 分钟组会讲法
- 0-3 min: 说明这篇论文处在 AI 自动科研链条的哪一层: literature、search、agent、tool、code、benchmark 或 discovery。
- 3-8 min: 讲背景和核心问题，说明为什么原有 LLM/RAG/agent 方法不够。
- 8-16 min: 拆系统或 benchmark 设计，画出模块、输入输出、评分器或反馈回路。
- 16-23 min: 讲实验和证据，尤其区分“任务完成”与“科研价值”。
- 23-27 min: 讲局限、失败模式、与已有 01-30 篇的关系。
- 27-30 min: 抛出讨论问题，落到本课题组可做的 follow-up。

## 背景和核心问题
自动科研 agent 必须边想边做：读论文、查资料、跑代码、观察结果，再调整计划。纯 chain-of-thought 只会想，纯 action policy 又缺少解释。

## 摘要级内容抓手
根据摘要，这篇工作的核心关注点可以概括为：While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate...

这一段不是替代精读，而是帮组会开场快速定位：标题和摘要显示，这篇工作应被放在“reasoning + acting 基础范式”这一层，而不是泛泛当作又一个 LLM 应用。

## 方法拆解
组会重点讲 reasoning trace 与 action trace 如何交替，以及这种结构如何支持工具调用、错误恢复和可解释决策。

组会时建议画一张三层图：
1. 输入层: 用户问题、论文库、网页、代码库、数据集或交互环境。
2. 决策层: planner、retriever、reasoner、tool caller、critic、evaluator 或 memory。
3. 输出层: 报告、代码 patch、实验结果、benchmark 分数、hypothesis 或可验证 artifact。

## 证据与实验怎么看
实验通常展示在问答、网页或交互任务上优于单纯推理/单纯行动。你要强调它给 AI Scientist 提供的是行动循环骨架。

读实验部分时不要只摘最高分。要记录作者证明了什么、没有证明什么，以及这个证明是否能支撑“AI 自动科研”这个更大的 claim。

## 局限、风险和批判点
ReAct 风格容易受上下文长度和错误观察影响；它并不保证长期计划和科学判断。

对老师汇报时可以主动讲一个反例：如果把这篇方法直接用于真实科研，最可能在哪一步失败，是检索、评价、工具调用、实验设计，还是 novelty 判断？

## 放在 70 篇图谱中的位置
这篇补的是 reasoning + acting 基础范式 这一块能力。它不一定直接产生科学发现，但它支撑 AI Scientist 的某个必要环节。

与已有 01-30 篇的关系: 01-09 更偏端到端 AI Scientist，10-17 偏文献/idea/novelty，18-24 偏科学 agent benchmark，25-30 偏领域发现和可验证发现。本篇应作为这些主线的补充能力或评测依据。

## 组会讨论问题
- 这篇论文解决的是 AI 自动科研链条中的哪一环，能否独立构成 PhD 方向？
- 它的评价指标是否真的衡量了科研质量，还是只衡量任务完成度？
- 如果迁移到 video generation / world models，应替换哪些工具、数据和 verifier？

## 可复现或可延伸的 follow-up
把文献调研流程写成 ReAct loop：Thought 是当前 gap，Action 是检索/阅读/实验，Observation 是证据。

## 建议 slides
1. 标题 + 本文属于哪一层能力。
2. 为什么需要这篇: 它补齐 70 篇图谱中的哪块缺口。
3. 方法/benchmark 总图。
4. 关键实验或任务设计。
5. 和 01-30 中最相近论文的对比。
6. 局限和失败模式。
7. 对 video generation / world models / AI Scientist 方向的启发。
8. 讨论问题和下一步可做实验。
