# auto-research 74 篇 · 按类型情况简报（Catalog by Type）

> **这份文档是什么**：把 [`papers/`](../papers/) 里全部 **74 篇** auto-research 论文**按类型分九组**，逐组说明
> 「这组在回答什么问题、有几篇、整体成熟度/争议在哪」，再逐篇给一句话状态。
>
> **和 [`README.md`](README.md) 的区别**：README 是**链接索引**（按组列表 + 一句话，便于跳转去读报告）；
> 本文是**按类型的情况分析**——多了「流水线角色 / 体量 / 成熟度判断 / 机构·venue 标签 / 这一组的争议焦点」这几层。
> 想读单篇 → 点链接进对应报告；想先建判断 → 读本文。
>
> 标注约定：⭐⭐/⭐ = Tier-S 锚点；🆕 = 第二批新增（含 **Why 三连** + 强制 **★ 对我们的启发** 专节）；
> 🏛️ = 顶刊（如 Nature MI）。建设全程见 [`PROGRESS.md`](PROGRESS.md)，规范见 [`_STYLE-GUIDE.md`](_STYLE-GUIDE.md) /
> [`_STYLE-GUIDE-v2-why-and-inspiration.md`](_STYLE-GUIDE-v2-why-and-inspiration.md)。

---

## 全局总览

| 组 | 主题 | 篇数 | 体量 | 在科研流水线里的角色 |
|---|---|---:|---:|---|
| **A** | 综述 / 全景 | 8 | ~38.8 万 | 建坐标系（先读） |
| **B** | 端到端 AI Scientist | 13 | ~61.5 万 | 旗舰系统：把全流程跑通 |
| **C** | 创意 / 假设生成 | 10 | ~50.3 万 | 科研第一步：出点子 |
| **D** | Deep Research / 综述合成 | 8 | ~37.7 万 | 读文献、写带引用综述 |
| **E** | 评测 / Benchmark | 17 | ~77.8 万 | 方法论护城河：怎么算"做到了" |
| **F** | 自我改进 / 算法发现 | 8 | ~35.8 万 | 系统改自己、自动发现算法 |
| **G** | 批判 / 陷阱 | 5 | ~23.6 万 | 判断力护城河（PhD 必读） |
| **H** | 前沿 / 生态 | 2 | ~10.1 万 | 2025–2026 共演化愿景 |
| **I** | 域内落地发现 | 3 | ~14.1 万 | 落到化学/生物/材料的真发现 |
| | **合计** | **74** | **~349.8 万字符** | 第一批 40 + 第二批 🆕 34；单篇 24k–68k |

> **读法建议**：A（坐标系）→ B 的 [2408.06292](2408.06292-ai-scientist-v1.md)（0 号系统）→ 按兴趣进 C/D/E/F →
> I 组看真实学科发现 → 最后用 G 组建立判断力。

---

## A. 综述 / 全景（8 篇，~38.8 万字符）——先读，建坐标系

**这组回答**：这个领域整体长什么样、有哪些流派、用什么坐标衡量。
**情况**：6 篇第一批 + 2 篇新增，时间跨度 2025.03→2025.12，**越新覆盖越系统**。第一篇 [2505.13259](2505.13259-survey-automation-to-autonomy.md) 的「Tool→Analyst→Scientist 三级自主性阶梯」是全库公认坐标系，其余综述各有侧重（deep research 专题 / 跨学科全景 / 科研生命周期）。读完这组即可给任何一篇论文"定位"。

| 论文 (arXiv) | 报告 | 一句话情况 |
|---|---|---|
| ⭐ From Automation to Autonomy (2505.13259) | [报告](2505.13259-survey-automation-to-autonomy.md) | Tool→Analyst→Scientist 三级自主性阶梯，**全库坐标系** |
| Survey of LLM Scientific Agents (2503.24047) | [报告](2503.24047-survey-scientific-intelligence-agents.md) | 科研 agent 区别于通用 agent 的关键能力 + 路线图 |
| Deep Research Agents Roadmap (2506.18096) | [报告](2506.18096-survey-deep-research-agents-roadmap.md) | "deep research"这一支的系统综述 |
| AI4Research (2507.01903) | [报告](2507.01903-survey-ai4research.md) | 覆盖最广的跨学科 AI-for-science 全景 |
| Deep Research Autonomous (2508.12752) | [报告](2508.12752-survey-deep-research-autonomous.md) | 自主研究 agent 专题综述 |
| Deep Research Systematic (2512.02038) | [报告](2512.02038-survey-deep-research-systematic.md) | 目前**最新**的系统综述 |
| 🆕 Scientific LLMs: Data→Agent (2508.21148) | [报告](2508.21148-survey-scientific-llms-data-to-agent.md) | 从数据基础到 agent 前沿**最全分类法** |
| 🆕 LLM4SR (2501.04306) | [报告](2501.04306-survey-llm4sr.md) | 按"假设→实验→写作→评审"科研生命周期综述 |

---

## B. 端到端 AI Scientist（13 篇，~61.5 万字符）——旗舰系统

**这组回答**：能不能把"选题→实验→写论文"整条流水线全自动跑通。
**情况**：全库第二大组、旗舰最密集。[2408.06292](2408.06292-ai-scientist-v1.md)（Sakana v1）是领域起点也是 **v1 标杆范文**；v2 用树搜索拿下"首个过人类评审的 AI 论文"。新增 5 篇把边界往外推：Google co-scientist 与 FutureHouse **Robin 已做出真实生物发现（ripasudil 治干性 AMD）**，MLE-STAR 在 Kaggle 夺牌 64%。**争议焦点**：这组最容易"宣称>实测"，必须和 G 组（批判）对照读——很多"全自动论文"经不起独立复现。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| ⭐⭐ AI Scientist v1 (2408.06292) | [报告](2408.06292-ai-scientist-v1.md) | 领域起点，五阶段全自动、每篇 <$15（**v1 标杆**） | Sakana |
| ⭐ AI Scientist-v2 (2504.08066) | [报告](2504.08066-ai-scientist-v2-tree-search.md) | 去模板+树搜索；**首个过人类评审**的 AI 论文 | Sakana |
| ⭐ AI co-scientist (2502.18864) | [报告](2502.18864-google-ai-co-scientist.md) | 多 agent 生成-辩论-进化 + 湿实验验证 | Google |
| Agent Laboratory (2501.04227) | [报告](2501.04227-agent-laboratory.md) | PhD/Postdoc/Professor 多角色 + 人在环 | |
| AI-Researcher (2505.18705) | [报告](2505.18705-ai-researcher-hkuds.md) | 全自动管线 + 自提 Scientist-Bench | HKUDS |
| NovelSeek/InternAgent (2505.16938) | [报告](2505.16938-novelseek-internagent.md) | 12 个科研任务闭环，真涨点 | |
| AgentRxiv (2503.18102) | [报告](2503.18102-agentrxiv-collaborative.md) | 给 agent 的预印本服务器，成果可累积 | |
| AIGS (2411.11910) | [报告](2411.11910-aigs-automated-falsification.md) | 把"可证伪性"放进闭环 | |
| 🆕 CycleResearcher (2411.00816) | [报告](2411.00816-cycleresearcher-automated-review.md) | 研究-评审-精修闭环，CycleReviewer 当奖励 | ICLR'25 |
| 🆕 ⭐ Robin (2505.13400) | [报告](2505.13400-robin-futurehouse-discovery.md) | 多 agent 端到端**真发现 ripasudil 治干性 AMD** | FutureHouse |
| 🆕 URSA (2506.22653) | [报告](2506.22653-ursa-universal-research-agent.md) | 通用"研究与科学 agent"框架 | |
| 🆕 DeepScientist (2509.26603) | [报告](2509.26603-deepscientist-frontier-findings.md) | 渐进逼近并宣称超越前沿（**全库最长 68k**） | |
| 🆕 ⭐ MLE-STAR (2506.15692) | [报告](2506.15692-mle-star-google-ml-engineering.md) | 搜 SOTA+定向精修，MLE-bench Lite 夺牌 64% | Google |

---

## C. 创意 / 假设生成（10 篇，~50.3 万字符）——科研第一步

**这组回答**：AI 能不能出真正新颖、且可行的科研点子。
**情况**：这组有一对**必须对读的"打架"论文**——[2409.04109](2409.04109-can-llms-generate-novel-ideas.md)（大规模人类对照：AI 点子**更新颖**）vs [2506.20803](2506.20803-ideation-execution-gap.md)（真去执行后**反而不如人**）。结论：novelty ≠ feasibility，自动出点子容易、出"能落地的好点子"难。新增 5 篇引入结构化方法（知识图谱、社区模拟、老虎机式数据驱动），把"灵感"变成可搜索的过程。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| ⭐ Can LLMs Generate Novel Ideas? (2409.04109) | [报告](2409.04109-can-llms-generate-novel-ideas.md) | 首个大规模人类对照：AI 点子更新颖、可行性略低 | Stanford |
| ⭐ The Ideation-Execution Gap (2506.20803) | [报告](2506.20803-ideation-execution-gap.md) | **真去执行后，AI 点子反而不如人**（泼冷水） | |
| ResearchAgent (2404.07738) | [报告](2404.07738-researchagent-iterative-ideation.md) | 学术图谱 + ReviewingAgents 迭代精化 | |
| Combinatorial Creativity (2412.14141) | [报告](2412.14141-llm-combinatorial-creativity.md) | 把创意建模为"组合式创造" | |
| Deep Ideation (2511.02238) | [报告](2511.02238-deep-ideation-concept-network.md) | 在科学概念网络上生成新点子 | |
| 🆕 SciAgents (2409.05556) | [报告](2409.05556-sciagents-graph-reasoning.md) | 本体知识图谱 + 多智能体图推理 | MIT |
| 🆕 ResearchTown (2412.17767) | [报告](2412.17767-researchtown-community-simulator.md) | 研究社区建成 agent-数据图 + TextGNN | ICML'25 |
| 🆕 HypoGeniC (2404.04326) | [报告](2404.04326-hypogenic-hypothesis-generation.md) | UCB 老虎机式数据驱动假设生成 + 错误样本库 | |
| 🆕 Many Heads (2410.09403) | [报告](2410.09403-many-heads-multi-agent-ideation.md) | 多智能体多视角协作提升创意 | |
| 🆕 AutoSDT (2506.08140) | [报告](2506.08140-autosdt-data-driven-discovery.md) | 自动造"数据驱动发现"任务，喂养开放 co-scientist | |

---

## D. Deep Research / 综述合成（8 篇，~37.7 万字符）

**这组回答**：能不能自己读海量文献、写出带准确引用的长综述。
**情况**：成熟度相对高、**最接近能用**的一支。[STORM](2402.14207-storm-wikipedia-from-scratch.md) 开源可跑，[OpenScholar](2411.14199-openscholar-ai2.md) 引用准确率比肩专家。新增 4 篇把重心从"检索拼接"推到"**用 RL 在真实 web 上端到端训练**"（DeepResearcher / WebThinker / WebSailor）。核心指标是**引用忠实度**——这也是我们 m9.4 模块复现的重点。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| ⭐ STORM (2402.14207) | [报告](2402.14207-storm-wikipedia-from-scratch.md) | 多视角提问+检索→带引用长综述；**开源可跑** | Stanford |
| Co-STORM (2408.15232) | [报告](2408.15232-co-storm-unknown-unknowns.md) | 人机协作 + 动态思维导图，发现"未知的未知" | |
| ⭐ OpenScholar (2411.14199) | [报告](2411.14199-openscholar-ai2.md) | 4500 万论文库；引用准确率比肩专家 | Ai2 |
| DeepScholar-Bench (2508.20033) | [报告](2508.20033-deepscholar-bench.md) | 评测"生成式研究综述"的 live benchmark | |
| 🆕 ⭐ PaperQA2 (2409.13740) | [报告](2409.13740-paperqa2-superhuman-synthesis.md) | 超人类文献综合，RAG 改造成多步 agent | FutureHouse |
| 🆕 ⭐ DeepResearcher (2504.03160) | [报告](2504.03160-deepresearcher-rl-realworld.md) | **首个真实 web 环境端到端 RL 训练** | EMNLP'25 |
| 🆕 ⭐ WebThinker (2504.21776) | [报告](2504.21776-webthinker-deep-research.md) | 推理模型"思考内"自主搜索+深浏览+起草 | NeurIPS'25 |
| 🆕 WebSailor (2507.02592) | [报告](2507.02592-websailor-web-agent.md) | 不确定性削减+DUPO 攻克超难 web 推理 | 阿里通义 |

---

## E. 评测 / Benchmark（17 篇，~77.8 万字符）——最大的一组，方法论护城河

**这组回答**：凭什么说 agent"做到了"？用什么任务、什么指标、能不能被刷。
**情况**：**全库最大组**，且这正是这个领域真正的护城河——不是"又造了个 agent"，而是"你凭什么相信它"。这组集中了 OpenAI（PaperBench/MLE-bench）、METR（RE-Bench）、Ai2（DiscoveryBench/SUPER/AstaBench）、Meta（MLGym）等最权威出品。**最值得记的数字是一串"低分"**：DiscoveryBench 最强仅 25%、SUPER 16.3%、EXP-Bench **可执行成功率 0.5%**、ResearchCodeBench <40%——它们冷静地标出了"全自动科研"离可靠还有多远。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| ⭐ PaperBench (2504.01848) | [报告](2504.01848-paperbench-openai.md) | 从论文复现，8316 个可打分子任务 | OpenAI |
| ⭐ MLE-bench (2410.07095) | [报告](2410.07095-mle-bench-openai.md) | 75 个 Kaggle 赛测端到端 ML 工程 | OpenAI |
| ⭐ RE-Bench (2411.15114) | [报告](2411.15114-re-bench-metr.md) | 7 环境 61 专家对照；**时间预算决定人机谁赢** | METR |
| ScienceAgentBench (2410.05080) | [报告](2410.05080-scienceagentbench.md) | 44 篇论文抽 102 个数据驱动任务 | |
| MLGym (2502.14499) | [报告](2502.14499-mlgym-meta.md) | 首个 ML 研究 Gym 环境，可训 RL agent | Meta |
| MLR-Bench (2505.19955) | [报告](2505.19955-mlr-bench.md) | 开放式 ML 研究评测 | |
| MLAgentBench (2310.03302) | [报告](2310.03302-mlagentbench.md) | 最早把 agent 放进 ML 实验流的**奠基基准** | |
| AstaBench (2510.21652) | [报告](2510.21652-astabench-ai2.md) | 科研 agent 全套件，成本受控评测 | Ai2 |
| CORE-Bench (2409.11363) | [报告](2409.11363-core-bench-reproducibility.md) | 从论文代码+数据测可复现性 | |
| SciCode (2407.13168) | [报告](2407.13168-scicode-benchmark.md) | 科学家出题的真实科研代码生成 | |
| 🆕 DiscoveryBench (2407.01725) | [报告](2407.01725-discoverybench-data-driven.md) | 从数据搜索+验证假设，**最强系统仅 25%** | Ai2 |
| 🆕 SUPER (2409.07440) | [报告](2409.07440-super-research-repositories.md) | 测"把研究仓库配起来跑通"，GPT-4o 仅 16.3% | Ai2 |
| 🆕 EXP-Bench (2505.24785) | [报告](2505.24785-exp-bench-ai-experiments.md) | 完整实验全流程，**可执行成功率仅 0.5%** | |
| 🆕 ResearchCodeBench (2506.02314) | [报告](2506.02314-researchcodebench-novel-code.md) | 实现最新论文新代码，最好 <40% | NeurIPS'25 |
| 🆕 BixBench (2503.00096) | [报告](2503.00096-bixbench-computational-biology.md) | 计算生物学真实分析 agent 基准 | FutureHouse |
| 🆕 InnovatorBench (2510.27598) | [报告](2510.27598-innovatorbench-llm-research.md) | 评测"创新性 LLM 研究"能力 | |
| 🆕 HypoBench (2504.11524) | [报告](2504.11524-hypobench-hypothesis-benchmark.md) | 系统化、原则化地评测假设生成 | |

---

## F. 自我改进 / 自动算法发现（8 篇，~35.8 万字符）

**这组回答**：系统能不能改写自己、并自动发现新算法/新数学。
**情况**：最"科幻"也最硬核的一支。[AlphaEvolve](2506.13131-alphaevolve-deepmind.md)（DeepMind，**v2 标杆**）进化整份代码库，**48 次乘法打破 Strassen 保持 56 年的纪录**，是全组的高光；SEAL（MIT）让 LLM **永久改自身权重**。**关键分水岭**：凡 fitness 能被**机器自动验证**的（数学/代码/跑分）就可信、就强；凡 fitness 靠自评的就直接埋 reward hacking 的雷——这正是我们 m9.7 模块复现的"优化器背满泄漏集、holdout 死守 0.500"现象。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| ⭐ Darwin Gödel Machine (2505.22954) | [报告](2505.22954-darwin-godel-machine.md) | Agent 改写自己代码，**真实跑分**做适应度进化 | Sakana |
| ⭐ ADAS (2408.08435) | [报告](2408.08435-adas-agentic-system-design.md) | 让 agent 设计 agent，在代码空间搜索架构 | |
| 🆕 ⭐⭐ AlphaEvolve (2506.13131) | [报告](2506.13131-alphaevolve-deepmind.md) | 进化整份代码库，**48 乘破 Strassen 56 年**（**v2 标杆**） | DeepMind |
| 🆕 ⭐ SEAL (2506.10943) | [报告](2506.10943-seal-self-adapting-lms.md) | 自造微调数据+RL，让 LLM **永久改自身权重** | MIT |
| 🆕 Gödel Agent (2410.04444) | [报告](2410.04444-godel-agent-self-referential.md) | 运行时改写自身逻辑、递归自我改进 | ACL'25 |
| 🆕 LLM-SR (2404.18400) | [报告](2404.18400-llm-sr-equation-discovery.md) | 科学方程发现=用 LLM 写程序+进化 | ICLR'25 |
| 🆕 AI Mathematician (2505.22451) | [报告](2505.22451-ai-mathematician-frontier-math.md) | 朝全自动前沿数学研究 | |
| 🆕 CodeEvolve (2510.14150) | [报告](2510.14150-codeevolve-open-evolutionary.md) | AlphaEvolve 的开源复刻（岛屿 GA+加权 LLM 集成） | open |

---

## G. 批判 / 陷阱（5 篇，~23.6 万字符）——判断力护城河，PhD 必读

**这组回答**：这些"成功"里哪些是刷榜、伪造、幻觉？边界在哪？
**情况**：**全库最该和 B 组对读的一组**，篇数不多但权重极高。[Hidden Pitfalls](2509.08713-critique-hidden-pitfalls.md) 是"照妖镜"（刷榜/伪造数据集/幻觉结果）；[Why LLMs Aren't Scientists Yet](2601.03315-critique-why-not-scientists-yet.md)（2026.01，最新）总结四次真实尝试的教训。读完这组才有资格判断前八组任何一篇的"宣称"。

| 论文 (arXiv) | 报告 | 一句话情况 |
|---|---|---|
| Wishful Thinking? ARI (2502.14297) | [报告](2502.14297-critique-wishful-thinking-ari.md) | 对 Sakana AI Scientist 最系统的独立批判 |
| ⭐ Hidden Pitfalls (2509.08713) | [报告](2509.08713-critique-hidden-pitfalls.md) | 刷榜、伪造数据集、幻觉结果的**照妖镜** |
| Fail Without Implementation (2506.01372) | [报告](2506.01372-critique-fail-without-implementation.md) | 瓶颈在"可靠实现"而非"想点子" |
| ⭐ Why LLMs Aren't Scientists Yet (2601.03315) | [报告](2601.03315-critique-why-not-scientists-yet.md) | 四次真实尝试的教训（**最新批判**） |
| 🆕 Biomedical Acceleration Limits (2508.16613) | [报告](2508.16613-critique-biomedical-acceleration-limits.md) | 通用 AI 加速生物医学研究的边界 |

---

## H. 前沿 / 生态（2 篇，~10.1 万字符）——2025–2026 最新

**这组回答**：再往前一步，人机科研生态会长成什么样。
**情况**：愿景型、数量少但代表"下一站"。从单个 agent 升到**整个科研生态**（投稿-评审-发布、人机共演化）。

| 论文 (arXiv) | 报告 | 一句话情况 |
|---|---|---|
| OmniScientist (2511.16931) | [报告](2511.16931-omniscientist-coevolving.md) | 人机共演化的科研生态愿景 |
| 🆕 aiXiv (2508.15126) | [报告](2508.15126-aixiv-ai-scientist-ecosystem.md) | 给 AI 科学家的开放投稿-评审-发布生态 |

---

## I. 域内落地发现（3 篇，~14.1 万字符）🆕 第二批新增组

**这组回答**：抛开 benchmark，真在化学/生物/材料里做出了什么发现。
**情况**：第二批专门新增，补上"会做科研 vs 真做出发现"的最后一公里。**ChemCrow 发在 Nature MI** 是顶刊背书；SparksMatter 用**物理感知**给材料发现收口；Paper2Agent 把"论文+代码"变成可靠 agent。它们正好补上 AlphaEvolve「必须能自动验证」够不到的半张图——**湿实验 / 物理约束收口**。

| 论文 (arXiv) | 报告 | 一句话情况 | 出处 |
|---|---|---|---|
| 🆕 🏛️ ChemCrow (2304.05376) | [报告](2304.05376-chemcrow-chemistry-tools.md) | GPT-4 + 18 化学工具，自主合成与发现 | Nature MI'24 |
| 🆕 SparksMatter (2508.02956) | [报告](2508.02956-sparksmatter-materials-discovery.md) | 多智能体 + **物理感知**的无机材料自主发现 | MIT |
| 🆕 Paper2Agent (2509.06917) | [报告](2509.06917-paper2agent-reproducible-agents.md) | 把"论文+代码"自动变成可靠交互 agent | Stanford Zou |

---

## 跨组横切：三个看库的角度

### 1）按批次
- **第一批 40 篇**（2026-06-26）：奠定 A–H 八组骨架，规范 [`_STYLE-GUIDE.md`](_STYLE-GUIDE.md)，标杆 [2408.06292](2408.06292-ai-scientist-v1.md)。
- **第二批 🆕 34 篇**（2026-06-27）：老师批示"量≥70、更新更权威、解读更详尽"，全部 2025–2026 新作，**每篇多两维**：
  **Why 三连**（问题层/设计层/结果层）+ 强制 **`## ★ 对我们的启发`**；并新开 **I 组**。标杆 [2506.13131 AlphaEvolve](2506.13131-alphaevolve-deepmind.md)。

### 2）按权威出处（部分）
DeepMind（AlphaEvolve）、Google（co-scientist / MLE-STAR）、OpenAI（PaperBench / MLE-bench）、Meta（MLGym）、
METR（RE-Bench）、Ai2（OpenScholar / DiscoveryBench / SUPER / AstaBench）、MIT（SEAL / SciAgents / SparksMatter）、
FutureHouse（PaperQA2 / Robin / BixBench）、Stanford（STORM / Paper2Agent）、Sakana（AI Scientist v1/v2 / DGM）；
顶会 ICLR/ICML/NeurIPS/ACL/EMNLP 2025 一批 + Nature MI（ChemCrow）。

### 3）按"可信度怎么收口"（最重要的一刀）
| 收口方式 | 代表 | 可信度 |
|---|---|---|
| 机器自动验证（数学/代码/跑分） | AlphaEvolve、LLM-SR、DGM | **高** |
| 物理约束 / 湿实验 | Robin、SparksMatter、ChemCrow、co-scientist | **高** |
| 真实人类对照 / 真实 web | Can-LLMs-Ideas、RE-Bench、DeepResearcher | 中–高 |
| 仅自评 / 自审 | 多数纯生成式端到端系统 | **低（reward hacking 风险）** |

---

## 一句话收口（贯穿全 74 篇）

**自动科研已能跑通，但可信度最终全压在「独立验证」这一环上**：
凡能用**机器自动验证 / 物理约束 / 湿实验**收口的（AlphaEvolve 破 Strassen、Robin 发现 ripasudil、SparksMatter 物理感知）就更可信；
凡只能**自评 / 自审**的就埋着 reward hacking 的雷（G 组三面照妖镜 + 我们 m9.7/m9.8 模块亲手复现的"刷榜守恒"都在反复印证）。
E 组（17 篇评测）之所以是全库最大组，正是因为——**这个领域真正的护城河不是"又造了个 agent"，而是"你凭什么相信它"。**

---

> 配套：阅读指南 [`../README.md`](../README.md)｜教学系列 [`../CURRICULUM.md`](../CURRICULUM.md)｜可跑模块 `../m9.*/`｜
> 链接索引 [`README.md`](README.md)｜建设账本 [`PROGRESS.md`](PROGRESS.md)。**74/74 完成**（第一批 40 + 第二批 🆕 34）。
