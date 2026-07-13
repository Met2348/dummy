# 01. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

- 年份/来源: 2024 / arXiv
- 方向分类: end_to_end
- 本地论文: [01-ai-scientist.pdf](c:/Workspace/dummy/docs/literature/llm-auto-research/papers/01-ai-scientist.pdf)
- 在线来源: https://arxiv.org/abs/2408.06292
- 下载入口: https://arxiv.org/pdf/2408.06292

## 一句话定位
端到端 AI Scientist 方向的标志性起点，把 idea、代码实验、论文写作和自动 review 串成闭环。

## 30 分钟组会讲法
- 0-3 min: 用一句话说明这篇论文解决 AI 自动科研流程中的哪一环。
- 3-8 min: 讲背景和问题定义，说明旧方法或旧 workflow 卡在哪里。
- 8-16 min: 拆方法框架，重点画出模块、输入输出和反馈回路。
- 16-23 min: 讲实验或证据，区分作者真正证明了什么、没有证明什么。
- 23-27 min: 讲局限、风险和与其他论文的关系。
- 27-30 min: 抛出讨论问题，并落到我们能做的 follow-up。

## 背景和核心问题
问题不是 LLM 能否帮人写代码，而是能否把一个最小机器学习研究流程自动化。论文把科研拆成想法生成、新颖性检查、实验执行、论文写作和审稿反思，试图证明研究流程本身可以 agent 化。

## 方法拆解
系统从研究模板和 seed idea 出发，生成候选 idea，检索相关工作做 novelty check，修改代码库跑实验，再生成 LaTeX 论文并调用 reviewer agent 打分。关键工程取舍是把开放科研压缩到可执行的小型 ML 论文环境。

## 证据与实验怎么看
论文展示多个小领域中自动生成的论文和自动评审结果。强证据是完整流水线与大量样例；弱证据是科学价值主要靠自动评审和少量人工检查，离真实顶会创新仍有距离。

## 局限、风险和批判点
最大风险是 novelty illusion：系统会生成形式完整的论文，但贡献可能只是局部调参、概念拼接或语义重复。实验环境被强烈简化，自动 review 与专家审稿差距明显。

## 放在 LLM Auto Research / AI Scientist 图谱中的位置
这是所有后续 AI Scientist 系统的基准论文。读后续论文时可以问：它比 AI Scientist 多了什么，是更强搜索、更真实实验、更严格评测，还是更好的人机协作？

## 组会讨论问题
- 端到端自动化的最小可行科研单元应有多小？
- 自动 review 能否作为科研质量信号，还是只能做过滤器？
- 迁移到 video generation 时，最小实验环境如何设计？

## 可复现或可延伸的 follow-up
可复现一个极小版：给定 toy video diffusion repo，让 agent 生成 temporal coherence 改进 idea、跑实验、写短报告。

## 建议 slides
1. 标题 + 一句话定位。
2. 研究流程图: 这篇覆盖 literature / ideation / experiment / evaluation / writing 中的哪些环节。
3. 方法模块图。
4. 关键实验或案例。
5. 与相邻论文对比。
6. 局限和失败模式。
7. 对我们方向的启发。
8. 讨论问题和下一步实验。
