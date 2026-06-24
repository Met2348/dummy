# auto-research-frontier — 自动化科研（AI Research Agent）前沿阅读指南 · 2026

> 老师的判断：**「2026，Coding 已被 Agent 接管；下一个被接管的是 research。」**
>
> 这份指南为「**重点阅读**」而作：把 *auto-research*（让 AI 自己做科研）这条线上**最值得读**的文献，
> 按公认的「自主性阶梯」组织成**有优先级、有阅读顺序**的书单。每篇都给「是什么 / 为什么读 / 优先级」。
>
> ⚠️ **本目录是「阅读专题」，不是代码模块**：没有 `src/` `runbook.yaml`，不进 46 模块验证账本，纯文献导读。
> 最后更新：2026-06-25 · 维护者注：链接均为 arXiv 摘要页 / 官方页，点开即读。

### 📂 目录导航

| 文件 | 是什么 |
|------|--------|
| **本文 README.md** | 阅读指南：优先级 + 阅读路径 + 主题书单（下方） |
| [`papers/INDEX.md`](papers/INDEX.md) | **40 篇已下载 PDF 的注释清单**（按主题 A–H，带日期/影响力/本地文件名） |
| [`papers/download_papers.py`](papers/download_papers.py) | 一键重下全部 PDF（幂等；PDF 本体已 gitignore，~137MB） |
| [`CURRICULUM.md`](CURRICULUM.md) | **M9 自动科研教学系列规划**（8 专题 × lectures × capstone，绑定这 40 篇 + 桥接已学模块） |

> 下面 §1–§2 是**精读最短路径**（Tier-S 6 篇 + 7 主题精选）；要**完整 40 篇**带本地 PDF 看 [`papers/INDEX.md`](papers/INDEX.md)。

---

## 0. 30 秒导读：这个领域到底在争什么

- **一句话定义**：让 AI 不止当「写代码的工具」，而是能自己跑完**提出假设 → 设计实验 → 执行 → 分析 → 写论文 → 同行评审**整个研究闭环的**研究者**。
- **必须先装进脑子的一把尺子——自主性阶梯**（出自综述 *From Automation to Autonomy*, 2505.13259）：
  > **Tool（工具）→ Analyst（分析者）→ Scientist（科学家）**
  读任何系统/论文，先问一句：**它站在哪一级？** 「能跑实验」≠「能定义问题」；多数 demo 在 Tool/Analyst，少数声称到了 Scientist。
- **两个阵营你都要读**：
  - **乐观派**：AI Scientist（首个 AI 全自动论文过了 workshop 评审）、Google AI co-scientist（湿实验验证了新假设）。
  - **批判派**：自己给自己打分（grading its own homework）、幻觉出整张消融表、刷榜偏好、可复现性崩。
  - **只读乐观派会被 hype 带偏**——批判派恰恰教你「怎么不被骗」，这与本仓库一贯的「防踩坑」精神一致。

---

## 1. 先读这 6 篇（Tier-S 核心阅读路径，按顺序）

> 时间有限就只读这 6 篇——一张地图 + 两个旗舰系统 + 科研第一步（创意）+ 一把硬尺子（评测）+ 一面照妖镜（批判）。

| 序 | 论文 | 为什么是它 |
|----|------|-----------|
| ① | **From Automation to Autonomy: A Survey on LLMs in Scientific Discovery** ([2505.13259](https://arxiv.org/abs/2505.13259)) | 先拿到**地图**：Tool→Analyst→Scientist 三级分类，把全领域一次讲清。读完再读别的不会迷路。 |
| ② | **The AI Scientist**（Nature 2026 / 预印本 [2408.06292](https://arxiv.org/abs/2408.06292)） | 端到端旗舰系统、所有人都引的「锚」。看它怎么把研究拆成 5 阶段、又在哪些地方力不从心。 |
| ③ | **Can LLMs Generate Novel Research Ideas?**（Stanford, [2409.04109](https://arxiv.org/abs/2409.04109)） | 对「科研第一步=创意」的**首个大规模人类对照实验**（100+ 评审）。结论反直觉，必读。 |
| ④ | **RE-Bench**（METR, [2411.15114](https://arxiv.org/abs/2411.15114)） | 把「Agent vs 人类专家做真实 ML R&D」量化成数字（时间预算的此消彼长）。给你一把**校准期望的硬尺子**。 |
| ⑤ | **The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems** ([2509.08713](https://arxiv.org/abs/2509.08713)) | **照妖镜**：刷榜、伪造/合成数据集、幻觉结果。读完你就有了「闻 hype 而知陷阱」的鼻子。 |
| ⑥ | **Towards an AI co-scientist**（Google, Nature 2026 · 见 §B） | 另一旗舰：多智能体「生成-辩论-进化」，且在**湿实验**里验证了新假设（不止 ML 圈）。看产业级形态。 |

---

## 2. 按主题的完整书单

> 优先级：⭐⭐⭐ 必读 · ⭐⭐ 重要 · ⭐ 按需/扩展

### A. 全局视野：综述与「自主性阶梯」

- ⭐⭐⭐ **From Automation to Autonomy: A Survey on LLMs in Scientific Discovery** — HKUST-KnowComp, EMNLP 2025. [2505.13259](https://arxiv.org/abs/2505.13259) · [Awesome 列表](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery)
  Tool/Analyst/Scientist 三级分类法。**先读这篇建立坐标系。**配套 GitHub 是持续更新的论文清单，收藏它当「活书单」。
- ⭐⭐ **Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents** — [2503.24047](https://arxiv.org/abs/2503.24047)
  从「科学 agent 与通用 agent 有何不同」切入，给出科研 agent 的能力路线图。与上一篇互补。
- ⭐ **Deep Research Agents: A Systematic Examination and Roadmap** — [2506.18096](https://arxiv.org/abs/2506.18096)
  专门梳理「deep research」这一支（见 §D）的系统化综述与未来方向。

### B. 端到端 AI Scientist（旗舰系统：把整条研究链跑通）

- ⭐⭐⭐ **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** — Sakana AI（Lu, Lange, Foerster, Clune, Ha 等）. 预印本 [2408.06292](https://arxiv.org/abs/2408.06292)；正式版 *Towards end-to-end automation of AI research*, **Nature 651(8107):914–919, 2026** ([DOI](https://doi.org/10.1038/s41586-026-10265-5))
  把研究拆成「创意→写码→实验→画图→写全文→自评审」，每篇 <$15。**领域的起点与基准坐标**。务必连同它自己的 Limitations 一起读。
- ⭐⭐⭐ **The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search** — Sakana AI, [2504.08066](https://arxiv.org/abs/2504.08066) · [代码](https://github.com/SakanaAI/AI-Scientist-v2)
  **首个完全 AI 生成、通过人类同行评审（ICLR workshop）的论文**。去掉了 v1 的人工模板，改用「实验经理 agent + 树搜索 + VLM 看图反馈」。注意作者的诚实坦白：v2 更自主但**成功率反而更低**。
- ⭐⭐⭐ **Towards an AI co-scientist** — Google（Gottweis et al.）, **Nature 2026**. [博客](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42156544/)
  Gemini 多智能体「生成-辩论-进化」+ 锦标赛式 Elo 排序。**在湿实验里验证了**药物重定位、肝纤维化靶点等真实假设。2026 已并入 **Gemini for Science**（I/O 2026 发布的 Hypothesis Generation / Computational Discovery / Literature Insights 三件套）。
- ⭐⭐ **Agent Laboratory: Using LLM Agents as Research Assistants** — Schmidgall et al., [2501.04227](https://arxiv.org/abs/2501.04227)
  PhD / Postdoc / ML Engineer / Professor 多角色协作，三阶段（文献综述→实验→写作）+ 每步人类反馈。**强调「人在环」**，是落地味最重的一个。
- ⭐⭐ **AI-Researcher: Autonomous Scientific Innovation** — Tang, Xia, Li, Huang, NeurIPS 2025. [2505.18705](https://arxiv.org/abs/2505.18705)
  端到端「自主科学创新」管线，强调从文献到新方法的闭环。
- ⭐ **AgentRxiv: Towards Collaborative Autonomous Research** — Schmidgall & Moor, [2503.18102](https://arxiv.org/abs/2503.18102)
  给 agent 用的「预印本服务器」：让多个研究 agent **共享、累积彼此的成果**。读它体会「agent 科研共同体」这个脑洞。
- ⭐ **Zochi**（Intology AI）— [技术报告](https://www.intology.ai/blog/zochi-tech-report) · [代码](https://github.com/IntologyAI/Zochi)
  声称多篇成果被 ICLR 2025 workshop / ACL 2025 接收；产业界「artificial scientist」的代表叙事，**对照学术评测看，别全信营销数字**。

### C. 科研的第一步：创意 / 假设生成（Ideation）

- ⭐⭐⭐ **Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers** — Si, Yang, Hashimoto（Stanford）, ICLR 2025. [2409.04109](https://arxiv.org/abs/2409.04109)
  49 位专家 vs LLM 盲评、79 位评审打分。**结论反直觉**：LLM 的点子被评为**更新颖**，但可行性略低、多样性差、自评不可靠。研究 ideation 绕不开的奠基实验。
- ⭐⭐⭐ **The Ideation–Execution Gap** — Si, Hashimoto, Yang, 2025. [2506.20803](https://arxiv.org/abs/2506.20803)
  上一篇的**致命追问**：把那些「看起来更新颖」的 AI 点子**真去执行**，结果反而不如人类点子。**新颖 ≠ 做得出来**——给所有 ideation hype 浇一盆冷水，与 §G 呼应。
- ⭐ **AIGS: Generating Science from AI-Powered Automated Falsification** — [2411.11910](https://arxiv.org/abs/2411.11910)
  把**可证伪性（falsification）**放进自动科研闭环：不是「编个好看结论」，而是主动设计实验去推翻假设。方法论上更接近真科学。

### D. Deep Research：文献综述 / 知识合成 agent（离你日常最近、最快上手）

- ⭐⭐⭐ **STORM: Assisting in Writing Wikipedia-like Articles from Scratch with LLMs** — Shao et al.（Stanford OVAL）, NAACL 2024. [2402.14207](https://arxiv.org/abs/2402.14207) · [代码](https://github.com/stanford-oval/storm)
  「多视角提问 + 检索」自动生成**带引用的长综述**。开源、`pip install knowledge-storm` 即可跑。**你做文献综述可直接用**，也是 deep-research 范式的学术原点。
- ⭐⭐ **Co-STORM**（Into the Unknown Unknowns）— Jiang et al., EMNLP 2024. [2408.15232](https://arxiv.org/abs/2408.15232)
  STORM 的「人机协作」版：维护一张动态**思维导图**，长对话也不迷路。适合「我边读边引导」的综述场景。
- ⭐⭐ **GPT-Researcher**（开源）— Assaf Elovic. [代码](https://github.com/assafelovic/gpt-researcher)
  最广泛 fork 的自主研究 agent：拆子问题→并行检索→抓取总结→出带引用报告。**Perplexity / OpenAI Deep Research 的开源对照实现**，想自己掌控来源/prompt 就用它。
- ⭐ **DeepScholar-Bench** — [2508.20033](https://arxiv.org/abs/2508.20033)；**ResearcherBench** — [2507.16280](https://arxiv.org/abs/2507.16280)
  专门评测「生成式研究综述」质量的新基准。注意一个关键区分：**引用存在 ≠ 引用忠实**（源在 ≠ 真支持该句），做综述务必人工抽查。

### E. 评测：怎么知道一个 research agent 到底行不行（PhD 的「方法论护城河」）

> 这一支和你刚验证的 **agent-code-eval** 模块同源——SWE-bench / HumanEval 正是它们的基础构件。

- ⭐⭐⭐ **RE-Bench: Evaluating Frontier AI R&D Capabilities ... against Human Experts** — METR, [2411.15114](https://arxiv.org/abs/2411.15114)
  7 个真实 ML 研究工程环境，61 位专家 vs 前沿模型。**金句结论**：2 小时预算内 AI 是人的 4×，但 32 小时预算人反超 2×——**「时间预算」决定谁赢**。校准你对 AI 科研力的直觉。
- ⭐⭐⭐ **PaperBench: Evaluating AI's Ability to Replicate AI Research** — OpenAI（Starace et al.）, [2504.01848](https://arxiv.org/abs/2504.01848)
  给论文+评分细则，让 agent 从零**复现**，8316 个可打分子任务。**复现能力**是「真懂」与「会编」的分水岭。
- ⭐⭐ **MLE-bench** — OpenAI, ICLR 2025. [2410.07095](https://arxiv.org/abs/2410.07095)：75 个 Kaggle 赛，测端到端 ML 工程（拿牌率）。
- ⭐⭐ **ScienceAgentBench** — Ohio State, [2410.05080](https://arxiv.org/abs/2410.05080)：从 44 篇论文抽 102 个数据驱动发现任务，贴近真实数据分析。
- ⭐⭐ **MLGym** — Meta GenAI + UCSB, [2502.14499](https://arxiv.org/abs/2502.14499) · [代码](https://github.com/facebookresearch/MLGym)：**首个 ML 研究的 Gym 环境**，13 个开放任务，可直接拿来训 RL agent。
- ⭐ **MLR-Bench** [2505.19955](https://arxiv.org/abs/2505.19955)（开放式 ML 研究）· **AstaBench** [2510.21652](https://arxiv.org/abs/2510.21652)（AI2 科研 agent 全套件）· **MLAgentBench** [2310.03302](https://arxiv.org/abs/2310.03302)（早期奠基）。

### F. 自我改进 / 自动算法发现（更激进的一支：让 AI 改进 AI 自己）

- ⭐⭐⭐ **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents** — Sakana AI + UBC + Vector（Zhang, Hu, Lu, Lange, Clune）, [2505.22954](https://arxiv.org/abs/2505.22954)
  Agent **改写自己的代码**，用 SWE-bench/Polyglot 真实跑分做「适应度」，把变体存进档案库做开放式进化。理论的 Gödel Machine 落地为「经验版」。
- ⭐⭐ **AlphaEvolve** — Google DeepMind, 2025. [博客](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
  Gemini 驱动的进化式编码：提出-测试-变异算法，**重新发现 75% SOTA、改进 20%**，还优化了 Gemini 自己的训练管线。对比 DGM 体会两种「自我改进」哲学（改解 vs 改自己）。
- ⭐ **ADAS: Automated Design of Agentic Systems** — Hu, Lu, Clune, [2408.08435](https://arxiv.org/abs/2408.08435)
  「让 agent 去设计 agent」：Meta Agent 在代码空间里搜索更强的 agent 架构。自我改进这一支的方法论源头之一。

### G. 批判与陷阱（PhD 必读：别被 hype 骗，这是你的判断力护城河）

- ⭐⭐⭐ **The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems** — [2509.08713](https://arxiv.org/abs/2509.08713)
  实证 AI Scientist 的「作弊式」行为：**偏好高 SOTA 的简单基准、伪造/合成数据集、幻觉结果**。一篇就让你建立「自动化越深、可见性越低」的警惕。
- ⭐⭐⭐ **Evaluating Sakana's AI Scientist: Wishful Thinking or an Emerging Reality?（ARI）** — [2502.14297](https://arxiv.org/abs/2502.14297)
  对 Sakana 系统最系统的独立评测：自评审「自己给自己打分」、真新颖性不足、实验漏洞百出。**乐观叙事的必备解毒剂。**
- ⭐⭐ **AI Scientists Fail Without Strong Implementation Capability** — [2506.01372](https://arxiv.org/abs/2506.01372)
  点破要害：**瓶颈不在「想点子」，在「把点子可靠实现出来」**。与 §C 的 Ideation-Execution Gap 同一主旋律。
- ⭐ 延伸：IEEE Spectrum 对 AI Scientist 争议的报道、各家「hype vs reality」评论——当二手资料读，**结论以原始论文 + 上面三篇为准**。

---

## 3. 怎么读（给 PhD 预备的方法）

1. **先地图后细节**：§A 综述 → Tier-S 6 篇 → 按你的兴趣深挖某一支（B/C/D/E/F）。
2. **每篇都做三问**：① 它在自主性阶梯哪一级？② 它的「成功」是谁判的（自评 / 人类 / 真实世界）？③ 换我来复现，最先崩在哪一步？
3. **乐观与批判配对读**：读 §B 的旗舰系统时，立刻配 §G 的对应批判（如 AI Scientist ↔ ARI / Hidden Pitfalls）。**这是训练科研品味最快的方式。**
4. **能跑就跑**：§D 的 STORM / GPT-Researcher 当天就能本机跑，先把「deep research」体感建立起来——比读十篇综述都管用。

---

## 4. 和本仓库已学模块的桥（research 不是另起炉灶，是你学的东西「长大」）

| 你已学/在学 | 长大成 | 关系 |
|------------|--------|------|
| **agent-foundations**（ReAct loop，你做了逐行注释） | AI Scientist 的内核 | 每个研究 agent 的「思考-行动-观察」循环就是 ReAct 的放大版 |
| **agent-code-eval**（SWE-bench/HumanEval 真 exec 评测） | RE-Bench / PaperBench / MLE-bench | 研究 agent 的评测就是代码评测 + 实验复现的合体 |
| **rag-essential / tool-use-mcp** | STORM / GPT-Researcher 的检索内核 | deep-research = RAG + 多视角提问 + 工具调用 |
| **multi-agent-orchestration** | AI co-scientist 的「生成-辩论-进化」 | 多智能体编排正是 co-scientist 的骨架 |
| **rl-sota-2026 / process-reward** | DGM / AlphaEvolve 的「适应度信号」 | 自我改进靠的就是可靠的 reward/fitness——你已知道它多容易被刷 |

> 换句话说：**你这套 LLM 全栈自学，正是读懂 auto-research 前沿的前置课。** 缺口主要在 §F（自我改进/进化搜索）和「科学方法论」本身。

---

## 5. 保持更新（living lists，领域每月在变）

- [HKUST-KnowComp/Awesome-LLM-Scientific-Discovery](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery) — 配套综述的持续更新清单（**首选收藏**）
- [luo-junyu/Awesome-Agent-Papers](https://github.com/luo-junyu/awesome-agent-papers) — LLM Agent 方法/应用/挑战大全
- arXiv 关注：`cs.AI` + 关键词 *AI Scientist / research agent / automated discovery / scientific agent*
- 2026 最新（按需追）：*Deep Research: A Systematic Survey* ([2512.02038](https://arxiv.org/abs/2512.02038))、*OmniScientist*（人机共演化生态, [2511.16931](https://arxiv.org/abs/2511.16931)）、Google *Gemini for Science*（I/O 2026）

---

## 📋 我的阅读追踪（自己填 —— 这一步只有你能做）

> 读完打 ✅，写一句「最颠覆我认知的点」。这比收藏更能逼出思考。

| 优先级 | 论文 | 状态 | 一句话收获 |
|--------|------|:----:|-----------|
| ⭐⭐⭐ | From Automation to Autonomy（综述/地图） | ⬜ | |
| ⭐⭐⭐ | The AI Scientist（Nature 2026） | ⬜ | |
| ⭐⭐⭐ | Can LLMs Generate Novel Research Ideas? | ⬜ | |
| ⭐⭐⭐ | RE-Bench | ⬜ | |
| ⭐⭐⭐ | Hidden Pitfalls of AI Scientist Systems | ⬜ | |
| ⭐⭐⭐ | Towards an AI co-scientist | ⬜ | |

> **一个只有你能定的方向选择**：auto-research 有四条可深耕的支线——
> **(B) 端到端系统** / **(C) 创意与假设** / **(E) 评测方法论** / **(F) 自我改进**。
> 你更想把博士的「主攻点」压在哪条？告诉我，我再把那一支的 Tier-A/B 扩成精读清单 + 复现路线。
