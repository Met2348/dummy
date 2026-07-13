# 25. Autonomous chemical research with large language models

- 年份/来源: 2023 / Nature
- 方向分类: domain_chemistry
- 本地论文: [25-coscientist.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/25-coscientist.pdf)
- 在线来源: https://www.nature.com/articles/s41586-023-06792-0
- 下载入口: https://www.nature.com/articles/s41586-023-06792-0.pdf

## 一句话定位
LLM 驱动自主化学研究的 Nature 代表作，展示 LLM + 工具 + 实验自动化潜力。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
自动科研在化学中很有吸引力，因为文献、合成规划、仪器控制和实验验证可以形成闭环。

## 方法拆解
系统把 LLM 与网络检索、文献理解、化学工具和自动化实验平台连接，规划反应、调用工具、生成实验方案，并执行部分实验。

## 证据与实验怎么看
论文展示系统在化学问题上的规划和实验执行案例，说明 LLM agent 可以进入真实实验室工作流。

## 局限、风险和批判点
安全、可控性和专家监督非常关键。化学实验错误成本高，系统不能只靠语言自信行动。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是 domain-specific AI Scientist 的早期强证据：真正自动发现通常必须接工具和实验世界。

## 组会讨论问题
- LLM 控制实验设备需要怎样的权限边界？
- 工具调用失败时系统如何恢复？
- domain tools 的提升来自知识还是行动能力？

## 可复现或可延伸的 follow-up
借鉴安全设计，为代码实验 agent 设置 sandbox、权限和完整操作记录。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
