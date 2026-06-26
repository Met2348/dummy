# 组会汇报文档库 · auto-research 40 篇精读（PPT 风格 · 50 分钟级）

> 为 [`../papers/`](../papers/) 里**全部 40 篇** auto-research 论文各写的一份**约 20 页、50 分钟组会汇报级**中文文档。
> 每篇都**真读了 PDF**：intention/why > how，数学细节全，**每个公式前先讲直觉再逐个定义符号**，
> setting/metrics/parameters 写全、指标给定义式，数字标注 §/Table/Eq 出处，诚实区分「论文宣称 vs 批判/局限」。
>
> 写作规范见 [`_STYLE-GUIDE.md`](_STYLE-GUIDE.md)；标杆范文是 [`2408.06292-ai-scientist-v1.md`](2408.06292-ai-scientist-v1.md)；建设过程见 [`PROGRESS.md`](PROGRESS.md)。

## 怎么用

- **组会主讲**：每篇按 20 页骨架组织，二级标题即「幻灯片」，配 `> 主讲提示`（该页该说什么）；文末有「组会讨论问题」「一页速记」。
- **建议顺序**：先读 A 组 [2505.13259](2505.13259-survey-automation-to-autonomy.md)（坐标系）→ B 组 [2408.06292](2408.06292-ai-scientist-v1.md)（0 号系统）→ 按兴趣进 C/D/E/F → 最后 G 批判建立判断力。
- **配套代码**：本库 `m9.*` 模块把这些论文的核心思想做成了可跑的缩小版（见每篇文末「在课里的位置」）。

---

## A. 综述 / 全景（先读，建坐标系）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| From Automation to Autonomy (2505.13259) ⭐ | [报告](2505.13259-survey-automation-to-autonomy.md) | Tool→Analyst→Scientist 三级自主性阶梯，全领域坐标系 |
| Survey of LLM Scientific Agents (2503.24047) | [报告](2503.24047-survey-scientific-intelligence-agents.md) | 科研 agent 区别于通用 agent 的关键能力 + 路线图 |
| Deep Research Agents Roadmap (2506.18096) | [报告](2506.18096-survey-deep-research-agents-roadmap.md) | 「deep research」这一支的系统综述与路线图 |
| AI4Research (2507.01903) | [报告](2507.01903-survey-ai4research.md) | 覆盖最广的跨学科 AI-for-science 全景 |
| Deep Research Autonomous (2508.12752) | [报告](2508.12752-survey-deep-research-autonomous.md) | 自主研究 agent 专题综述 |
| Deep Research Systematic (2512.02038) | [报告](2512.02038-survey-deep-research-systematic.md) | 目前最新的系统综述 |

## B. 端到端 AI Scientist（旗舰系统）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| The AI Scientist v1 (2408.06292) ⭐⭐ | [报告](2408.06292-ai-scientist-v1.md) | 领域起点：五阶段全自动、每篇 <$15（标杆范文） |
| AI Scientist-v2 (2504.08066) ⭐ | [报告](2504.08066-ai-scientist-v2-tree-search.md) | 去模板 + 树搜索；首个过人类评审的 AI 论文 |
| AI co-scientist (2502.18864) ⭐ | [报告](2502.18864-google-ai-co-scientist.md) | 多 agent 生成-辩论-进化 + 湿实验验证 |
| Agent Laboratory (2501.04227) | [报告](2501.04227-agent-laboratory.md) | PhD/Postdoc/Professor 多角色 + 人在环 |
| AI-Researcher (2505.18705) | [报告](2505.18705-ai-researcher-hkuds.md) | 全自动管线 + 提出 Scientist-Bench |
| NovelSeek/InternAgent (2505.16938) | [报告](2505.16938-novelseek-internagent.md) | 12 个科研任务闭环，真涨点 |
| AgentRxiv (2503.18102) | [报告](2503.18102-agentrxiv-collaborative.md) | 给 agent 的预印本服务器，成果累积 |
| AIGS (2411.11910) | [报告](2411.11910-aigs-automated-falsification.md) | 把可证伪性放进闭环 |

## C. 创意 / 假设生成（科研第一步）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Can LLMs Generate Novel Ideas? (2409.04109) ⭐ | [报告](2409.04109-can-llms-generate-novel-ideas.md) | 首个大规模人类对照：AI 点子更新颖、可行性略低 |
| The Ideation-Execution Gap (2506.20803) ⭐ | [报告](2506.20803-ideation-execution-gap.md) | 真去执行：AI 点子反而不如人 |
| ResearchAgent (2404.07738) | [报告](2404.07738-researchagent-iterative-ideation.md) | 学术图谱 + ReviewingAgents 迭代精化 |
| Combinatorial Creativity (2412.14141) | [报告](2412.14141-llm-combinatorial-creativity.md) | 把创意建模为「组合式创造」 |
| Deep Ideation on Concept Network (2511.02238) | [报告](2511.02238-deep-ideation-concept-network.md) | 在科学概念网络上生成新点子 |

## D. Deep Research / 文献综述合成

| 论文 | 报告 | 一句话 |
|------|------|--------|
| STORM (2402.14207) ⭐ | [报告](2402.14207-storm-wikipedia-from-scratch.md) | 多视角提问+检索→带引用长综述；开源可跑 |
| Co-STORM (2408.15232) | [报告](2408.15232-co-storm-unknown-unknowns.md) | 人机协作 + 动态思维导图，发现未知的未知 |
| OpenScholar (2411.14199) ⭐ | [报告](2411.14199-openscholar-ai2.md) | 4500 万论文库；引用准确率比肩专家 |
| DeepScholar-Bench (2508.20033) | [报告](2508.20033-deepscholar-bench.md) | 评测「生成式研究综述」的 live benchmark |

## E. 评测 / Benchmark（方法论护城河）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| PaperBench (2504.01848) ⭐ | [报告](2504.01848-paperbench-openai.md) | 从论文复现，8316 个可打分子任务 |
| MLE-bench (2410.07095) ⭐ | [报告](2410.07095-mle-bench-openai.md) | 75 个 Kaggle 赛测端到端 ML 工程 |
| RE-Bench (2411.15114) ⭐ | [报告](2411.15114-re-bench-metr.md) | 7 环境 61 专家对照；时间预算决定谁赢 |
| ScienceAgentBench (2410.05080) | [报告](2410.05080-scienceagentbench.md) | 44 篇论文抽 102 个数据驱动任务 |
| MLGym (2502.14499) | [报告](2502.14499-mlgym-meta.md) | 首个 ML 研究 Gym 环境，可训 RL agent |
| MLR-Bench (2505.19955) | [报告](2505.19955-mlr-bench.md) | 开放式 ML 研究评测 |
| MLAgentBench (2310.03302) | [报告](2310.03302-mlagentbench.md) | 最早把 agent 放进 ML 实验流的奠基基准 |
| AstaBench (2510.21652) | [报告](2510.21652-astabench-ai2.md) | 科研 agent 全套件，成本受控评测 |
| CORE-Bench (2409.11363) | [报告](2409.11363-core-bench-reproducibility.md) | 从论文代码+数据测可复现性 |
| SciCode (2407.13168) | [报告](2407.13168-scicode-benchmark.md) | 科学家出题的真实科研代码生成 |

## F. 自我改进 / 自动算法发现

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Darwin Gödel Machine (2505.22954) ⭐ | [报告](2505.22954-darwin-godel-machine.md) | Agent 改写自己代码，真实跑分做适应度进化 |
| ADAS (2408.08435) ⭐ | [报告](2408.08435-adas-agentic-system-design.md) | 让 agent 设计 agent，在代码空间搜索架构 |

## G. 批判 / 陷阱（PhD 必读：判断力护城河）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Wishful Thinking? ARI 批判 (2502.14297) | [报告](2502.14297-critique-wishful-thinking-ari.md) | 对 Sakana AI Scientist 最系统的独立批判 |
| Hidden Pitfalls (2509.08713) ⭐ | [报告](2509.08713-critique-hidden-pitfalls.md) | 刷榜、伪造数据集、幻觉结果——照妖镜 |
| Fail Without Implementation (2506.01372) | [报告](2506.01372-critique-fail-without-implementation.md) | 瓶颈在「可靠实现」而非「想点子」 |
| Why LLMs Aren't Scientists Yet (2601.03315) ⭐ | [报告](2601.03315-critique-why-not-scientists-yet.md) | 四次真实尝试的教训（最新批判） |

## H. 前沿追踪（2026 最新）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| OmniScientist (2511.16931) | [报告](2511.16931-omniscientist-coevolving.md) | 人机共演化的科研生态愿景 |

---

> **40/40 完成**。配套阅读指南见 [`../README.md`](../README.md)，教学系列见 [`../CURRICULUM.md`](../CURRICULUM.md)，
> 可跑模块见 `../m9.*/`。一句话收口（贯穿全 40 篇）：**自动科研已能跑通，但可信度最终压在「独立验证」这一环。**
