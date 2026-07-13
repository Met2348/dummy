# 04. AI-Researcher: Autonomous Scientific Innovation

- 年份/来源: 2025 / arXiv
- 方向分类: end_to_end
- 本地论文: [04-ai-researcher.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/04-ai-researcher.pdf)
- 在线来源: https://arxiv.org/abs/2505.18705
- 下载入口: https://arxiv.org/pdf/2505.18705

## 一句话定位
提出完整 autonomous scientific innovation 框架，并配套 Scientist-Bench 评测自动科研能力。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
自动科研缺少两个东西：一是可串起完整科研流程的系统，二是能判断系统是否真正有创新的 benchmark。

## 方法拆解
AI-Researcher 编排文献理解、假设生成、算法实现、实验运行、结果分析和论文写作。Scientist-Bench 包含 guided innovation 与 open-ended exploration，用已有 SOTA paper 做参照。

## 证据与实验怎么看
论文报告系统在实现成功率和论文质量上接近人类研究稿件的部分指标。重要贡献是把自动科研从 demo 拉到 benchmark 语境。

## 局限、风险和批判点
若 benchmark 以已有论文为参照，可能奖励再发现而非真创新。open-ended 任务的 novelty、importance 和 long-term impact 仍难评分。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
适合接在 AI Scientist-v2 后读：它强调评测与任务集，而不只是 pipeline 或 search。

## 组会讨论问题
- Scientist-Bench 是否会泄漏已有论文答案？
- 应评估最终论文，还是评估中间决策过程？
- 什么 benchmark 能适配 video/world-model 研究？

## 可复现或可延伸的 follow-up
设计 WorldModel-Scientist-Bench：给定经典 video paper，要求 agent 提出未测试 ablation 并实现。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
