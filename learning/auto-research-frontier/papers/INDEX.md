# 文献库索引 — auto-research（74 篇 PDF 注释清单）

> 这 40 篇是 [`download_papers.py`](download_papers.py) 下载的全部论文（PDF 已 gitignore，跑脚本即可重下）。
> 按主题 A–H 分组，与 [`../README.md`](../README.md)（阅读指南）和 [`../CURRICULUM.md`](../CURRICULUM.md)（教学系列）一一对应。
> 影响力/时效标记：🏛️ 奠基·高影响 · 🧭 综述 · 🔥 2025–2026 前沿 · ⚖️ 批判/反思
> 更新：2026-06-27（第二批扩充至 74 篇，详见文末「第二批扩充」）

---

## A. 综述 / 全景（先读，建坐标系）

| 标记 | 论文（arXiv） | 机构 / 时间 | 本地文件 · 一句话 |
|------|--------------|------------|------------------|
| 🧭🏛️ | [From Automation to Autonomy: A Survey on LLMs in Scientific Discovery](https://arxiv.org/abs/2505.13259) | HKUST-KnowComp, EMNLP 2025 / 2025-05 | `2505.13259-survey-automation-to-autonomy.pdf` · **Tool→Analyst→Scientist 三级自主性阶梯**，全领域地图，第一篇读 |
| 🧭 | [Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents](https://arxiv.org/abs/2503.24047) | 2025-03 | `2503.24047-survey-scientific-intelligence-agents.pdf` · 科研 agent 与通用 agent 何异 + 能力路线图 |
| 🧭🔥 | [Deep Research Agents: A Systematic Examination and Roadmap](https://arxiv.org/abs/2506.18096) | 2025-06 | `2506.18096-survey-deep-research-agents-roadmap.pdf` · 「deep research」这一支的系统综述 |
| 🧭🔥 | [AI4Research: A Survey of AI for Scientific Research](https://arxiv.org/abs/2507.01903) | 2025-07 | `2507.01903-survey-ai4research.pdf` · 覆盖最广的一篇（7.5MB），跨学科 |
| 🧭🔥 | [Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/abs/2508.12752) | 2025-08 | `2508.12752-survey-deep-research-autonomous.pdf` · 自主研究 agent 专题综述 |
| 🧭🔥 | [Deep Research: A Systematic Survey](https://arxiv.org/abs/2512.02038) | 2025-12 | `2512.02038-survey-deep-research-systematic.pdf` · 目前最新的系统综述 |

## B. 端到端 AI Scientist（旗舰系统）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🏛️ | [The AI Scientist (v1)](https://arxiv.org/abs/2408.06292) | Sakana AI, 2024-08 → **Nature 2026** | `2408.06292-ai-scientist-v1.pdf` · 领域起点：5 阶段全自动、每篇 <$15 |
| 🏛️🔥 | [The AI Scientist-v2: Agentic Tree Search](https://arxiv.org/abs/2504.08066) | Sakana AI, 2025-04 | `2504.08066-ai-scientist-v2-tree-search.pdf` · **首个通过人类同行评审的 AI 论文**；树搜索+实验经理 agent |
| 🏛️🔥 | [Towards an AI co-scientist](https://arxiv.org/abs/2502.18864) | Google, 2025-02 → **Nature 2026** | `2502.18864-google-ai-co-scientist.pdf` · Gemini 多智能体生成-辩论-进化；**湿实验验证真假设** |
| 🔥 | [Agent Laboratory: LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227) | Schmidgall et al., 2025-01 | `2501.04227-agent-laboratory.pdf` · PhD/Postdoc/Professor 多角色 + 人在环 |
| 🔥 | [AI-Researcher: Autonomous Scientific Innovation](https://arxiv.org/abs/2505.18705) | HKUDS, NeurIPS 2025 / 2025-05 | `2505.18705-ai-researcher-hkuds.pdf` · 全自动管线 + 提出 Scientist-Bench |
| 🔥 | [NovelSeek / InternAgent: 闭环 ASR](https://arxiv.org/abs/2505.16938) | Shanghai AI Lab, 2025-05 | `2505.16938-novelseek-internagent.pdf` · 12 个科研任务，假设→验证闭环，真涨点 |
| 🔥 | [AgentRxiv: Collaborative Autonomous Research](https://arxiv.org/abs/2503.18102) | Schmidgall & Moor, 2025-03 | `2503.18102-agentrxiv-collaborative.pdf` · 给 agent 用的「预印本服务器」，成果累积 |
| | [AIGS: Generating Science from Automated Falsification](https://arxiv.org/abs/2411.11910) | 2024-11 | `2411.11910-aigs-automated-falsification.pdf` · 把**可证伪性**放进闭环，方法论更严谨 |

## C. 创意 / 假设生成（科研第一步）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🏛️ | [Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109) | Stanford (Si/Yang/Hashimoto), ICLR 2025 / 2024-09 | `2409.04109-can-llms-generate-novel-ideas.pdf` · **首个大规模人类对照**：AI 点子更新颖、可行性略低 |
| 🔥⚖️ | [The Ideation-Execution Gap](https://arxiv.org/abs/2506.20803) | Stanford, 2025-06 | `2506.20803-ideation-execution-gap.pdf` · 上一篇的冷水：真去执行，AI 点子反而不如人 |
| | [ResearchAgent: Iterative Research Idea Generation](https://arxiv.org/abs/2404.07738) | KAIST/MSR (Baek), NAACL 2025 / 2024-04 | `2404.07738-researchagent-iterative-ideation.pdf` · 学术图谱 + ReviewingAgents 迭代精化 |
| | [LLMs can Realize Combinatorial Creativity](https://arxiv.org/abs/2412.14141) | 2024-12 | `2412.14141-llm-combinatorial-creativity.pdf` · 把创意建模为「组合式创造」 |
| 🔥 | [Deep Ideation on Scientific Concept Network](https://arxiv.org/abs/2511.02238) | 2025-11 | `2511.02238-deep-ideation-concept-network.pdf` · 在科学概念网络上生成新点子 |

## D. Deep Research / 文献综述合成（离日常最近）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🏛️ | [STORM: Writing Wikipedia-like Articles from Scratch](https://arxiv.org/abs/2402.14207) | Stanford OVAL (Shao), NAACL 2024 / 2024-02 | `2402.14207-storm-wikipedia-from-scratch.pdf` · 多视角提问+检索→带引用长综述；**开源可跑** |
| | [Co-STORM: Into the Unknown Unknowns](https://arxiv.org/abs/2408.15232) | Stanford OVAL, EMNLP 2024 / 2024-08 | `2408.15232-co-storm-unknown-unknowns.pdf` · 人机协作 + 动态思维导图 |
| 🔥 | [OpenScholar: Synthesizing Literature with RAG LMs](https://arxiv.org/abs/2411.14199) | Ai2+UW (Asai), 2024-11 → Nature 2026 | `2411.14199-openscholar-ai2.pdf` · 4500 万开放论文检索库；引用准确率比肩人类专家 |
| 🔥 | [DeepScholar-Bench](https://arxiv.org/abs/2508.20033) | 2025-08 | `2508.20033-deepscholar-bench.pdf` · 评测「生成式研究综述」的 live benchmark |

## E. 评测 / Benchmark（方法论护城河）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🔥 | [PaperBench: Replicate AI Research](https://arxiv.org/abs/2504.01848) | OpenAI, 2025-04 | `2504.01848-paperbench-openai.pdf` · 从论文复现，8316 个可打分子任务 |
| 🏛️ | [MLE-bench](https://arxiv.org/abs/2410.07095) | OpenAI, ICLR 2025 / 2024-10 | `2410.07095-mle-bench-openai.pdf` · 75 个 Kaggle 赛测端到端 ML 工程 |
| 🏛️ | [RE-Bench: Frontier AI R&D vs Human Experts](https://arxiv.org/abs/2411.15114) | METR, 2024-11 | `2411.15114-re-bench-metr.pdf` · 7 环境，61 专家对照；**时间预算决定谁赢** |
| | [ScienceAgentBench](https://arxiv.org/abs/2410.05080) | Ohio State, 2024-10 | `2410.05080-scienceagentbench.pdf` · 44 篇论文抽 102 个数据驱动任务 |
| 🔥 | [MLGym: Gym for AI Research Agents](https://arxiv.org/abs/2502.14499) | Meta+UCSB, 2025-02 | `2502.14499-mlgym-meta.pdf` · **首个 ML 研究 Gym 环境**，可训 RL agent |
| 🔥 | [MLR-Bench: Open-Ended ML Research](https://arxiv.org/abs/2505.19955) | 2025-05 | `2505.19955-mlr-bench.pdf` · 开放式 ML 研究评测 |
| 🏛️ | [MLAgentBench](https://arxiv.org/abs/2310.03302) | Stanford (Huang), ICML 2024 / 2023-10 | `2310.03302-mlagentbench.pdf` · 最早把 agent 放进 ML 实验流的奠基基准 |
| 🔥 | [AstaBench](https://arxiv.org/abs/2510.21652) | Ai2, 2025-10 | `2510.21652-astabench-ai2.pdf` · 科研 agent 全套件，严格基准 |
| | [CORE-Bench: Computational Reproducibility](https://arxiv.org/abs/2409.11363) | Princeton (Siegel), 2024-09 | `2409.11363-core-bench-reproducibility.pdf` · 从论文代码+数据测可复现性 |
| | [SciCode](https://arxiv.org/abs/2407.13168) | 2024-07 | `2407.13168-scicode-benchmark.pdf` · 科学家出题的真实科研代码生成 |

## F. 自我改进 / 自动算法发现（最激进的一支）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🔥 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Sakana/UBC/Vector, 2025-05 | `2505.22954-darwin-godel-machine.pdf` · Agent **改写自己代码**，用真实跑分做适应度做进化 |
| 🏛️ | [ADAS: Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435) | UBC (Hu/Lu/Clune), 2024-08 | `2408.08435-adas-agentic-system-design.pdf` · **让 agent 设计 agent**，在代码空间搜索架构 |
| 🔥 | **AlphaEvolve**（无 arXiv，未下载）| Google DeepMind, 2025-05 | [DeepMind 博客/白皮书](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) · 进化式编码，重发现 75% SOTA、改进 20% |

## G. 批判 / 陷阱（PhD 必读：判断力护城河）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| ⚖️ | [Evaluating Sakana's AI Scientist: Wishful Thinking?（ARI）](https://arxiv.org/abs/2502.14297) | 2025-02 | `2502.14297-critique-wishful-thinking-ari.pdf` · 对 Sakana 系统最系统的独立批判 |
| ⚖️🔥 | [The More You Automate, the Less You See: Hidden Pitfalls](https://arxiv.org/abs/2509.08713) | 2025-09 | `2509.08713-critique-hidden-pitfalls.pdf` · 刷榜、伪造数据集、幻觉结果——照妖镜 |
| ⚖️ | [AI Scientists Fail Without Strong Implementation Capability](https://arxiv.org/abs/2506.01372) | 2025-06 | `2506.01372-critique-fail-without-implementation.pdf` · 瓶颈在「可靠实现」而非「想点子」 |
| ⚖️🔥 | [Why LLMs Aren't Scientists Yet: Lessons from Four Attempts](https://arxiv.org/abs/2601.03315) | 2026-01 | `2601.03315-critique-why-not-scientists-yet.pdf` · 四次真实自主研究尝试的教训（最新批判） |

## H. 前沿追踪（2026 最新）

| 标记 | 论文 | 机构 / 时间 | 本地文件 · 一句话 |
|------|------|------------|------------------|
| 🔥 | [OmniScientist: Co-evolving Ecosystem of Human and AI Scientists](https://arxiv.org/abs/2511.16931) | 2025-11 | `2511.16931-omniscientist-coevolving.pdf` · 人机共演化的科研生态愿景 |

---

## 第二批扩充（🆕 34 篇，2026-06；老师要求量≥70、更新更权威）

> 精读报告同样新增「Why 三连 + ★对我们的启发(Inspires Us)」两维。下载脚本已含，跑脚本即重下。

| 组 | arXiv · 一句话 |
|----|----------------|
| A 综述 | `2508.21148` 科学 LLM：数据→agent 前沿全分类法 · `2501.04306` LLM4SR：按科研生命周期综述 |
| B 系统 | `2411.00816` CycleResearcher(ICLR'25) · `2505.13400` Robin⭐(FutureHouse,湿实验发现) · `2506.22653` URSA · `2509.26603` DeepScientist · `2506.15692` MLE-STAR⭐(Google) |
| C 创意 | `2409.05556` SciAgents(MIT,知识图谱) · `2412.17767` ResearchTown(ICML'25) · `2404.04326` HypoGeniC · `2410.09403` Many-Heads · `2506.08140` AutoSDT |
| D Deep Research | `2409.13740` PaperQA2⭐(FutureHouse) · `2504.03160` DeepResearcher⭐(EMNLP'25,RL) · `2504.21776` WebThinker⭐(NeurIPS'25) · `2507.02592` WebSailor(阿里) |
| E 评测 | `2407.01725` DiscoveryBench(Ai2) · `2409.07440` SUPER(Ai2) · `2505.24785` EXP-Bench · `2506.02314` ResearchCodeBench(NeurIPS'25) · `2503.00096` BixBench · `2510.27598` InnovatorBench · `2504.11524` HypoBench |
| F 自改进/算法发现 | `2506.13131` AlphaEvolve⭐⭐(DeepMind,破Strassen) · `2506.10943` SEAL⭐(MIT) · `2410.04444` Gödel-Agent(ACL'25) · `2404.18400` LLM-SR(ICLR'25) · `2505.22451` AI-Mathematician · `2510.14150` CodeEvolve |
| G 批判 | `2508.16613` 生物医学加速的边界 |
| H 生态 | `2508.15126` aiXiv：AI 科学家的投稿-评审-发布生态 |
| I 域内落地 | `2304.05376` ChemCrow🏛️(Nature MI) · `2508.02956` SparksMatter(MIT,材料) · `2509.06917` Paper2Agent(Stanford) |

---

## 统计

- **共 74 篇 PDF**（第一批 40 + 第二批 🆕 34）。AlphaEvolve 已有 arXiv 版 `2506.13131` 并纳入。
- 时间：2023–2024 奠基约 14 篇，2025 前沿约 56 篇，2026 最新若干；综述 8、批判/立场 5、域内落地 3。
- 权威分布：DeepMind/Google/MIT/Ai2/FutureHouse/Stanford + ICLR/ICML/NeurIPS/ACL/EMNLP 2025 / Nature(MI)。
- Virtual Lab（Nature 2025）仅 bioRxiv 反爬，未纳入；其余 74 篇均 arXiv 可一键重下。
- **重下/更新**：`python learning/auto-research-frontier/papers/download_papers.py`（幂等，新增论文加进 `PAPERS` 列表即可；支持第 4 元素给完整 URL）。
