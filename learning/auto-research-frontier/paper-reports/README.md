# 组会汇报文档库 · auto-research 74 篇精读（PPT 风格 · 50 分钟级）

> 为 [`../papers/`](../papers/) 里**全部 74 篇** auto-research 论文各写的一份**约 20 页、50 分钟组会汇报级**中文文档。
> 每篇都**真读了 PDF**：intention/why > how，数学细节全，**每个公式前先讲直觉再逐个定义符号**，
> setting/metrics/parameters 写全、指标给定义式，数字标注 §/Table/Eq 出处，诚实区分「论文宣称 vs 批判/局限」。
>
> **分两批建成**：第一批 40 篇（基础规范 [`_STYLE-GUIDE.md`](_STYLE-GUIDE.md)，标杆 [`2408.06292`](2408.06292-ai-scientist-v1.md)）；
> 第二批 34 篇（2026-06，老师要求量≥70、更新更权威，并**新增两维**：更强的 **Why 三连**（问题/设计/结果层）+ 强制
> **`## ★ 对我们的启发（Inspires Us）`** 专节）——见增强规范 [`_STYLE-GUIDE-v2-why-and-inspiration.md`](_STYLE-GUIDE-v2-why-and-inspiration.md)、
> v2 标杆 [`2506.13131 AlphaEvolve`](2506.13131-alphaevolve-deepmind.md)。建设全程见 [`PROGRESS.md`](PROGRESS.md)。
> 下方标 🆕 者为第二批新增。

## 怎么用

- **组会主讲**：每篇按 20 页骨架组织，二级标题即「幻灯片」，配 `> 主讲提示`（该页该说什么）；文末有「组会讨论问题」「一页速记」；第二批每篇还有 **★ 对我们的启发** 一节，直接给"我们下周能试什么"。
- **建议顺序**：先读 A 组 [2505.13259](2505.13259-survey-automation-to-autonomy.md)（坐标系）→ B 组 [2408.06292](2408.06292-ai-scientist-v1.md)（0 号系统）→ 按兴趣进 C/D/E/F → I 组看落到化学/生物/材料的真实发现 → 最后 G 批判建立判断力。
- **配套代码**：本库 `m9.*` 模块把这些论文的核心思想做成了可跑的缩小版（见每篇文末「在课里的位置 / 对我们的启发」）。

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

| 🆕 Scientific LLMs: Data→Agent Frontiers (2508.21148) | [报告](2508.21148-survey-scientific-llms-data-to-agent.md) | 从数据基础到 agent 前沿的最全分类法 |
| 🆕 LLM4SR (2501.04306) | [报告](2501.04306-survey-llm4sr.md) | 按科研生命周期（假设→实验→写作→评审）综述 |

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
| 🆕 CycleResearcher (2411.00816) | [报告](2411.00816-cycleresearcher-automated-review.md) | 研究-评审-精修闭环，CycleReviewer 当奖励（ICLR'25） |
| 🆕 Robin (2505.13400) ⭐ | [报告](2505.13400-robin-futurehouse-discovery.md) | 多 agent 端到端发现 ripasudil 治干性 AMD（FutureHouse） |
| 🆕 URSA (2506.22653) | [报告](2506.22653-ursa-universal-research-agent.md) | 通用「研究与科学 agent」框架 |
| 🆕 DeepScientist (2509.26603) | [报告](2509.26603-deepscientist-frontier-findings.md) | 渐进逼近并宣称超越前沿的自治系统 |
| 🆕 MLE-STAR (2506.15692) ⭐ | [报告](2506.15692-mle-star-google-ml-engineering.md) | 搜 SOTA 模型+定向精修，MLE-bench Lite 夺牌 64%（Google） |

## C. 创意 / 假设生成（科研第一步）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Can LLMs Generate Novel Ideas? (2409.04109) ⭐ | [报告](2409.04109-can-llms-generate-novel-ideas.md) | 首个大规模人类对照：AI 点子更新颖、可行性略低 |
| The Ideation-Execution Gap (2506.20803) ⭐ | [报告](2506.20803-ideation-execution-gap.md) | 真去执行：AI 点子反而不如人 |
| ResearchAgent (2404.07738) | [报告](2404.07738-researchagent-iterative-ideation.md) | 学术图谱 + ReviewingAgents 迭代精化 |
| Combinatorial Creativity (2412.14141) | [报告](2412.14141-llm-combinatorial-creativity.md) | 把创意建模为「组合式创造」 |
| Deep Ideation on Concept Network (2511.02238) | [报告](2511.02238-deep-ideation-concept-network.md) | 在科学概念网络上生成新点子 |
| 🆕 SciAgents (2409.05556) | [报告](2409.05556-sciagents-graph-reasoning.md) | 本体知识图谱+多智能体图推理（MIT Buehler） |
| 🆕 ResearchTown (2412.17767) | [报告](2412.17767-researchtown-community-simulator.md) | 把研究社区建成 agent-数据图+TextGNN（ICML'25） |
| 🆕 HypoGeniC (2404.04326) | [报告](2404.04326-hypogenic-hypothesis-generation.md) | UCB 老虎机式数据驱动假设生成+错误样本库 |
| 🆕 Many Heads (2410.09403) | [报告](2410.09403-many-heads-multi-agent-ideation.md) | 多智能体多视角协作提升创意生成 |
| 🆕 AutoSDT (2506.08140) | [报告](2506.08140-autosdt-data-driven-discovery.md) | 自动造数据驱动发现任务，喂养开放 co-scientist |

## D. Deep Research / 文献综述合成

| 论文 | 报告 | 一句话 |
|------|------|--------|
| STORM (2402.14207) ⭐ | [报告](2402.14207-storm-wikipedia-from-scratch.md) | 多视角提问+检索→带引用长综述；开源可跑 |
| Co-STORM (2408.15232) | [报告](2408.15232-co-storm-unknown-unknowns.md) | 人机协作 + 动态思维导图，发现未知的未知 |
| OpenScholar (2411.14199) ⭐ | [报告](2411.14199-openscholar-ai2.md) | 4500 万论文库；引用准确率比肩专家 |
| DeepScholar-Bench (2508.20033) | [报告](2508.20033-deepscholar-bench.md) | 评测「生成式研究综述」的 live benchmark |
| 🆕 PaperQA2 (2409.13740) ⭐ | [报告](2409.13740-paperqa2-superhuman-synthesis.md) | 超人类文献综合，RAG 改造为多步 agent（FutureHouse） |
| 🆕 DeepResearcher (2504.03160) ⭐ | [报告](2504.03160-deepresearcher-rl-realworld.md) | 首个真实 web 环境端到端 RL 训练（EMNLP'25） |
| 🆕 WebThinker (2504.21776) ⭐ | [报告](2504.21776-webthinker-deep-research.md) | 推理模型在「思考内」自主搜索+深浏览+起草（NeurIPS'25） |
| 🆕 WebSailor (2507.02592) | [报告](2507.02592-websailor-web-agent.md) | 不确定性削减+DUPO 攻克超难 web 推理（阿里通义） |

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
| 🆕 DiscoveryBench (2407.01725) | [报告](2407.01725-discoverybench-data-driven.md) | 从数据搜索+验证假设，最强系统仅 25%（Ai2） |
| 🆕 SUPER (2409.07440) | [报告](2409.07440-super-research-repositories.md) | 测「把研究仓库配起来跑通」，GPT-4o 仅 16.3%（Ai2） |
| 🆕 EXP-Bench (2505.24785) | [报告](2505.24785-exp-bench-ai-experiments.md) | 完整研究实验全流程，可执行成功率仅 0.5% |
| 🆕 ResearchCodeBench (2506.02314) | [报告](2506.02314-researchcodebench-novel-code.md) | 实现最新论文新代码，最好 <40%（NeurIPS'25） |
| 🆕 BixBench (2503.00096) | [报告](2503.00096-bixbench-computational-biology.md) | 计算生物学真实分析 agent 基准（FutureHouse） |
| 🆕 InnovatorBench (2510.27598) | [报告](2510.27598-innovatorbench-llm-research.md) | 评测「创新性 LLM 研究」能力 |
| 🆕 HypoBench (2504.11524) | [报告](2504.11524-hypobench-hypothesis-benchmark.md) | 系统化、原则化地评测假设生成 |

## F. 自我改进 / 自动算法发现

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Darwin Gödel Machine (2505.22954) ⭐ | [报告](2505.22954-darwin-godel-machine.md) | Agent 改写自己代码，真实跑分做适应度进化 |
| ADAS (2408.08435) ⭐ | [报告](2408.08435-adas-agentic-system-design.md) | 让 agent 设计 agent，在代码空间搜索架构 |
| 🆕 AlphaEvolve (2506.13131) ⭐⭐ | [报告](2506.13131-alphaevolve-deepmind.md) | 进化整份代码库，48 次乘法破 Strassen 56 年（DeepMind；v2 标杆） |
| 🆕 SEAL (2506.10943) ⭐ | [报告](2506.10943-seal-self-adapting-lms.md) | 自造微调数据+RL，让 LLM 永久改自身权重（MIT） |
| 🆕 Gödel Agent (2410.04444) | [报告](2410.04444-godel-agent-self-referential.md) | 运行时改写自身逻辑、递归自我改进（ACL'25） |
| 🆕 LLM-SR (2404.18400) | [报告](2404.18400-llm-sr-equation-discovery.md) | 科学方程发现=用 LLM 写程序+进化（ICLR'25） |
| 🆕 AI Mathematician (2505.22451) | [报告](2505.22451-ai-mathematician-frontier-math.md) | 朝全自动前沿数学研究 |
| 🆕 CodeEvolve (2510.14150) | [报告](2510.14150-codeevolve-open-evolutionary.md) | AlphaEvolve 的开源复刻（岛屿 GA+加权 LLM 集成） |

## G. 批判 / 陷阱（PhD 必读：判断力护城河）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| Wishful Thinking? ARI 批判 (2502.14297) | [报告](2502.14297-critique-wishful-thinking-ari.md) | 对 Sakana AI Scientist 最系统的独立批判 |
| Hidden Pitfalls (2509.08713) ⭐ | [报告](2509.08713-critique-hidden-pitfalls.md) | 刷榜、伪造数据集、幻觉结果——照妖镜 |
| Fail Without Implementation (2506.01372) | [报告](2506.01372-critique-fail-without-implementation.md) | 瓶颈在「可靠实现」而非「想点子」 |
| Why LLMs Aren't Scientists Yet (2601.03315) ⭐ | [报告](2601.03315-critique-why-not-scientists-yet.md) | 四次真实尝试的教训（最新批判） |
| 🆕 Biomedical Acceleration Limits (2508.16613) | [报告](2508.16613-critique-biomedical-acceleration-limits.md) | 通用 AI 加速生物医学研究的边界在哪 |

## H. 前沿追踪 / 生态（2025–2026 最新）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| OmniScientist (2511.16931) | [报告](2511.16931-omniscientist-coevolving.md) | 人机共演化的科研生态愿景 |
| 🆕 aiXiv (2508.15126) | [报告](2508.15126-aixiv-ai-scientist-ecosystem.md) | 给 AI 科学家的开放投稿-评审-发布生态 |

## I. 域内落地发现（🆕 新增组：把"会做科研"落到化学/生物/材料的真实发现）

| 论文 | 报告 | 一句话 |
|------|------|--------|
| 🆕 ChemCrow (2304.05376) 🏛️ | [报告](2304.05376-chemcrow-chemistry-tools.md) | GPT-4+18 化学工具，自主合成与发现（Nature MI'24） |
| 🆕 SparksMatter (2508.02956) | [报告](2508.02956-sparksmatter-materials-discovery.md) | 多智能体+物理感知的无机材料自主发现（MIT） |
| 🆕 Paper2Agent (2509.06917) | [报告](2509.06917-paper2agent-reproducible-agents.md) | 把论文+代码自动变成可靠交互 agent（Stanford Zou） |

---

> **74/74 完成**（第一批 40 + 第二批 🆕 34）。配套阅读指南见 [`../README.md`](../README.md)，教学系列见 [`../CURRICULUM.md`](../CURRICULUM.md)，
> 可跑模块见 `../m9.*/`。一句话收口（贯穿全 74 篇）：**自动科研已能跑通，但可信度最终压在「独立验证」这一环**——
> 凡能"机器自动验证/物理约束/湿实验"做收口的（AlphaEvolve、Robin、SparksMatter）就更可信，凡只能"自评/自审"的就埋着 reward hacking 的雷。
