# 08. Robin: A multi-agent system for automating scientific discovery

- 年份/来源: 2025 / arXiv
- 方向分类: domain_biomed
- 本地论文: [08-robin.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/08-robin.pdf)
- 在线来源: https://arxiv.org/abs/2505.13400
- 下载入口: https://arxiv.org/pdf/2505.13400

## 一句话定位
FutureHouse 的 lab-in-the-loop 生物医学发现系统，完成从假设到实验解释的关键智力步骤。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
Robin 不满足于 ML toy benchmark，而是选择 dry age-related macular degeneration，尝试提出治疗策略、候选药物和后续机制分析。

## 方法拆解
系统整合文献搜索 agent 和数据分析 agent，流程包括背景调研、假设生成、实验设计、实验结果解释和更新假设。关键是与真实实验闭环连接。

## 证据与实验怎么看
论文称 Robin 提出增强 RPE phagocytosis 的治疗策略，识别并验证 ripasudil，并分析 RNA-seq 得到 ABCA1 机制线索。主文假设、实验计划、分析和图由 Robin 生成。

## 局限、风险和批判点
领域特定性很强，需要实验室、数据和专家共同支撑。自主发现比例和人类设定搜索空间的影响需要细看记录。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是 AI Scientist 从自动写 AI paper 走向真实湿实验科学的代表。可信自动发现通常必须接入外部实验世界。

## 组会讨论问题
- lab-in-the-loop 中人类 intervention 如何透明记录？
- 药物发现 novelty 是新 target、新用途还是新机制？
- 类似框架能否迁移到 robot/world model 实验闭环？

## 可复现或可延伸的 follow-up
类比 Robin：world-model agent 提出假设，仿真或 robot 实验验证，再用结果更新假设。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
