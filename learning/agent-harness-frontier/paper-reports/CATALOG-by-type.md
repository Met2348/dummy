# agent-harness 74 篇 · 按类型情况简报（Catalog by Type）

> **这份文档是什么**：把 [`papers/`](../papers/) 里全部 **74 篇** agent-harness 论文**按类型分八组**，逐组说明
> "这组在回答什么问题、有几篇、整体成熟度/争议在哪"，再逐篇给一句话状态。
>
> **和 [`README.md`](README.md) 的区别**：README 是**链接索引**（按组列表 + 一句话，便于跳转去读报告）；
> 本文是**按类型的情况分析**——多了"这一层在 harness 里的角色 / 体量 / 成熟度判断 / canon-前沿时间坐标 / 这一组的争议焦点"这几层。
> 想读单篇 → 点链接进对应报告；想先建判断 → 读本文。
>
> 标注约定：⭐⭐/⭐ = 库内核心锚点；六层分类 **E**nvironment/**T**ools/**C**ontext/**L**oop/**O**bservability/**V**alidation
> 是每篇报告标题行标注的主打 harness 层。建设全程见 [`PROGRESS.md`](PROGRESS.md)，规范见
> [`_STYLE-GUIDE-harness.md`](_STYLE-GUIDE-harness.md)。

---

## 全局总览

| 组 | 主题 | 篇数 | 主打层 | 体量 | 在 harness 里的角色 |
|---|---|---:|---|---:|---|
| **A** | 综述 / 框架与定义 | 8 | 全层 | ~45.0 万字符 | 建坐标系（先读） |
| **B** | 控制循环 / 推理-行动范式 | 10 | L 层 | ~43.6 万字符 | Agent 每一步"想什么、做什么"的骨架 |
| **C** | 工具接口 / ACI | 8 | T 层 | ~48.0 万字符 | Agent 如何正确、可靠地调用外部能力 |
| **D** | 上下文工程 / 记忆 | 16（最大组） | C 层 | ~90.8 万字符 | 有限上下文窗口下如何"记得住、用得对" |
| **E** | 编码 / SWE Agent 集成系统 | 10 | E/T 层 | ~62.6 万字符 | 把控制循环+工具接口整合成完整产品 |
| **F** | Web / 计算机使用 / GUI Agent | 7 | E/T 层 | ~39.0 万字符 | 在真实/仿真数字环境里执行任务 |
| **G** | Harness 评测 / scaffold-aware eval | 9 | O/V 层 | ~59.4 万字符 | 怎么公允地测出"harness 到底行不行" |
| **H** | 可靠性 / 安全 / 可观测 / 沙箱 | 6 | V 层 | ~29.5 万字符 | 出了问题怎么防、怎么查、怎么救 |
| | **合计** | **74** | E/T/C/L/O/V 全覆盖 | **~420.8 万字符** | 单篇 25.9k–91.8k，平均约 56.9k |

> **读法建议**：A（坐标系）→ B 的 [2210.03629 ReAct](2210.03629-react-reasoning-and-acting.md)（控制循环祖先）→
> 按兴趣进 C/D/E/F（能力构建的四根支柱）→ G 组 [2605.27922 Harness-Bench](2605.27922-harness-bench-measuring-harness-effects.md)（把全库论点量化实证）→
> 最后用 H 组建立"harness 会怎么出问题、怎么兜底"的判断力。

---

## A. 综述 / 框架与定义（8 篇，~45.0 万字符）——先读，建坐标系

**这组回答**：Harness 这个概念到底指什么、有哪些主流分类法、上下文工程/记忆这类子话题各自的全景是什么样。
**情况**：8 篇覆盖"定义 harness""编码 agent 脚手架分类学""架构-应用-评测三位一体""上下文工程综述""记忆综述""agent 综述奠基作""外化设计原则综述"——**没有单一的"全库坐标系"锚点论文**（不像 auto-research 有 2505.13259 那样的公认权威），而是几篇各有侧重的综述互补拼出全景，这本身也反映了 agent-harness 是一个比"auto-research"更年轻、尚未形成统一分类共识的领域。2308.11432 是这组里时间最早（2023）、被后续论文引用最多的奠基性综述。

| 论文 (arXiv) | 报告 | 一句话情况 |
|---|---|---|
| Natural Language Agent Harnesses (2603.25723) | [报告](2603.25723-natural-language-agent-harnesses.md) | "Harness"概念本身的定义与分类，**给全库定调** |
| Recreate: Experience-Driven Domain Agents (2601.11100) | [报告](2601.11100-recreate-experience-driven-domain-agents.md) | 从经验积累构建领域专属 agent |
| Inside the Scaffold: Coding Agent Taxonomy (2604.03515) | [报告](2604.03515-inside-the-scaffold-coding-agent-taxonomy.md) | 编码 agent 脚手架系统分类学，**本组最长（84k）** |
| AI Agent Systems: Architectures/Applications/Evaluation (2601.01743) | [报告](2601.01743-ai-agent-systems-architectures-applications-evaluation.md) | 架构-应用-评测三位一体综述 |
| Survey of Context Engineering for LLMs (2507.13334) | [报告](2507.13334-survey-of-context-engineering-for-llms.md) | D 组（记忆/上下文）的理论前置综述 |
| Memory in the Age of AI Agents (2512.13564) | [报告](2512.13564-memory-in-the-age-of-ai-agents.md) | Agent 记忆机制最新系统综述 |
| Survey on LLM-based Autonomous Agents (2308.11432) ⭐ | [报告](2308.11432-survey-on-llm-based-autonomous-agents.md) | **时间最早**（2023）、引用最广的奠基综述 |
| Externalization in LLM Agents: A Review (2604.08224) | [报告](2604.08224-externalization-in-llm-agents-review.md) | "把认知过程外化"设计原则综述 |

---

## B. 控制循环 / 推理-行动范式（10 篇，含 5 篇 canon 脊柱，~43.6 万字符）——L 层

**这组回答**：Agent 每一步到底该"想"还是"做"，想和做之间怎么交替，多步推理怎么组织成搜索。
**情况**：全库 canon 密度**最高**的一组——ReAct/Reflexion/ToT/Inner-Monologue/Self-Refine 五篇均为 2022–2023 年顶会 canon，共同构成"现代 agent 控制循环"的地基。LATS 把 ToT 和 MCTS 融合、接入真实环境反馈；后三篇（Modular Harness/Structured Graphs/FlowSteer）代表 2025–2026 前沿把线性循环升级为图结构、强化学习编排的最新尝试。**这组几乎是"读懂 Claude Code/任何现代 coding agent 循环设计"的必读前置**。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| ReAct (2210.03629) ⭐canon | [报告](2210.03629-react-reasoning-and-acting.md) | 推理+行动交错的控制循环**祖先** | canon 2022 |
| Reflexion (2303.11366) ⭐canon | [报告](2303.11366-reflexion-verbal-reinforcement-learning.md) | 语言反思代替梯度更新的自我改进 | canon 2023 |
| Tree of Thoughts (2305.10601) ⭐canon | [报告](2305.10601-tree-of-thoughts.md) | 推理建模为可搜索的树 | canon 2023 |
| Language Agent Tree Search / LATS (2310.04406) | [报告](2310.04406-language-agent-tree-search-lats.md) | ToT+MCTS+真实环境反馈融合 | 2023 |
| Plan-and-Solve Prompting (2305.04091) | [报告](2305.04091-plan-and-solve-prompting.md) | 先规划后执行的两阶段范式 | 2023 |
| Inner Monologue (2207.05608) ⭐canon | [报告](2207.05608-inner-monologue-embodied-planning.md) | 具身智能体内心独白式规划 | canon 2022 |
| Self-Refine (2303.17651) ⭐canon | [报告](2303.17651-self-refine-iterative-self-feedback.md) | 无需训练的迭代自我反馈精修 | canon 2023 |
| General Modular Harness / Gaming Agents (2507.11633) | [报告](2507.11633-general-modular-harness-gaming-agents.md) | 通用模块化 harness 在游戏场景验证 | 前沿 2025 |
| From Agent Loops to Structured Graphs (2604.11378) | [报告](2604.11378-from-agent-loops-to-structured-graphs.md) | 控制循环从线性到图结构 | 前沿 2026 |
| FlowSteer: Reinforced Workflow Orchestration (2602.01664) | [报告](2602.01664-flowsteer-reinforced-workflow-orchestration.md) | 强化学习编排工作流控制策略 | 前沿 2026 |

---

## C. 工具接口 / Agent-Computer Interface（8 篇，~48.0 万字符）——T 层

**这组回答**：Agent 怎么"正确地"调用外部工具——接口该怎么设计、怎么训练模型学会调用、怎么防止调用出错。
**情况**：SWE-agent 提出的 ACI（Agent-Computer Interface）概念是这组、也是全库 T 层的锚点术语；Toolformer 是"LLM 自学工具调用"这条技术路线的起点；ToolLLM/Gorilla 代表"喂海量真实 API"的规模化路线；Less-is-More 提出了与"越多越好"直觉相反的发现——**工具裁剪反而提升准确率**，这正是 2024–2025 年"Vercel 工具裁剪 80%→100%"现象背后的学术验证。FuncBenchGen 补上了"怎么公允评测工具调用、不被数据污染"这一评测方法论环节。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| SWE-agent: ACI (2405.15793) ⭐canon | [报告](2405.15793-swe-agent-agent-computer-interface.md) | **ACI 概念锚点**，T 层术语起点 | canon 2024 |
| Toolformer (2302.04761) ⭐canon | [报告](2302.04761-toolformer-self-taught-tool-use.md) | LLM 自学何时/如何调用工具 | canon 2023 |
| ToolLLM / ToolBench (2307.16789) | [报告](2307.16789-toolllm-toolbench-16000-apis.md) | 16000+ 真实 API 规模化工具学习 | 2023 |
| Gorilla (2305.15334) | [报告](2305.15334-gorilla-llm-connected-massive-apis.md) | 检索增强的海量 API 调用 | 2023 |
| ToolACE: Winning Function Calling (2409.00920) | [报告](2409.00920-toolace-winning-function-calling.md) | 高质量合成数据驱动函数调用训练 | 2024 |
| Less is More: Function Calling at the Edge (2411.15399) | [报告](2411.15399-less-is-more-function-calling-edge.md) | 工具裁剪提升准确率，**Vercel 现象学术版** | 2024 |
| MemTool: Short-Term Memory for Tool Calling (2507.21428) | [报告](2507.21428-memtool-short-term-memory-tool-calling.md) | 工具调用场景短期记忆机制 | 前沿 2025 |
| FuncBenchGen: Contamination-Free Eval (2509.26553) | [报告](2509.26553-funcbenchgen-contamination-free-eval.md) | 防数据污染的函数调用评测生成 | 前沿 2025 |

---

## D. 上下文工程 / 记忆（16 篇，最大组，~90.8 万字符）——C 层

**这组回答**：上下文窗口有限，agent 长程运行必然要面对"记什么、忘什么、怎么检索"——这组是全库体量最大、内部路线分歧最明显的一组。
**情况**：MemGPT 借鉴 OS 虚拟内存分页思想是这组的 canon 起点；Mem0/MEM1/Mem-α/A-MEM/MemAgent 等 10+ 篇代表 2024–2026 年记忆系统的快速迭代，路线包括"外挂向量记忆库"（Mem0）、"记忆与推理联合训练"（MEM1）、"卡片盒笔记法"（A-MEM，**与本库自己的 `MEMORY.md` `[[link]]` 双向链接机制同源**）、"主动折叠而非被动截断"（AgentFold）、"把管理记忆本身建模为一种动作"（Memory-as-Action）等。**这组内部存在明确的路线之争**：绝大多数论文主张"设计更好的记忆机制"，但压轴的 Less-Context-Better-Agents（2606.10209）作为反方压舱石，用实证挑战"记忆/上下文越丰富越好"这一默认假设——同批次前沿论文 AgentSwing（F 组 2603.27490）某种意义上是对这场路线之争的一个整合性回应：不再选边站队，而是主张"动态路由多种策略"。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| MemGPT: LLMs as Operating Systems (2310.08560) ⭐canon | [报告](2310.08560-memgpt-llms-as-operating-systems.md) | OS 虚拟内存分页思想搬进 LLM 记忆 | canon 2023 |
| MemoryBank (2305.10250) | [报告](2305.10250-memorybank-long-term-memory.md) | 长期记忆+遗忘曲线机制 | 2023 |
| Mem0: Production Long-Term Memory (2504.19413) | [报告](2504.19413-mem0-production-long-term-memory.md) | 生产级长期记忆系统 | 前沿 2025 |
| MEM1: Synergize Memory & Reasoning (2506.15841) | [报告](2506.15841-mem1-synergize-memory-reasoning.md) | 记忆与推理协同训练 | 前沿 2025 |
| Mem-α: RL Memory Construction (2509.25911) | [报告](2509.25911-mem-alpha-rl-memory-construction.md) | 强化学习学习如何构建记忆 | 前沿 2025 |
| A-MEM: Agentic Memory / Zettelkasten (2502.12110) | [报告](2502.12110-a-mem-agentic-memory-zettelkasten.md) | 卡片盒笔记法，**与我们 MEMORY.md 同源** | 前沿 2025 |
| MemAgent: Multi-Conv RL Memory (2507.02259) | [报告](2507.02259-memagent-multiconv-rl-memory.md) | 跨多轮对话强化学习记忆管理 | 前沿 2025 |
| IterResearch: Interaction Scaling (2511.07327) | [报告](2511.07327-iterresearch-interaction-scaling.md) | 迭代式交互规模化记忆策略 | 前沿 2025 |
| AgentFold: Proactive Context Folding (2510.24699) | [报告](2510.24699-agentfold-proactive-context-folding.md) | 主动折叠上下文而非被动截断 | 前沿 2025 |
| ACON: Context Compression for Agents (2510.00615) | [报告](2510.00615-acon-context-compression-agents.md) | 面向 agent 场景上下文压缩 | 前沿 2025 |
| Memory as Action: Context Curation (2510.12635) | [报告](2510.12635-memory-as-action-context-curation.md) | "管理记忆"本身建模为动作 | 前沿 2025 |
| MemSearcher: Reason-Search-Manage (2511.02805) | [报告](2511.02805-memsearcher-reason-search-manage-memory.md) | 推理-搜索-记忆管理一体化 | 前沿 2025 |
| AgentOCR: Optical Self-Compression (2601.04786) | [报告](2601.04786-agentocr-optical-self-compression.md) | "视觉渲染再 OCR"极限压缩 | 前沿 2026 |
| Agentic Memory: Unified LTM+STM (2601.01885) | [报告](2601.01885-agentic-memory-unified-ltm-stm.md) | 长短期记忆统一框架 | 前沿 2026 |
| RE-TRAC: Recursive Trajectory Compression (2602.02486) | [报告](2602.02486-re-trac-recursive-trajectory-compression.md) | 递归式轨迹压缩 | 前沿 2026 |
| Less Context, Better Agents (2606.10209) | [报告](2606.10209-less-context-better-agents.md) | **反方压舱石**：挑战"越多越好"默认假设 | 前沿 2026 |

---

## E. 编码 / SWE Agent 集成系统（10 篇，~62.6 万字符）——E/T 层

**这组回答**：把控制循环（B）+ 工具接口（C）+ 上下文管理（D）整合起来，能不能造出一个真正好用的编码 agent 产品。
**情况**：CodeAct 是这组的 canon 支点——"用可执行代码而非结构化 JSON 统一动作空间"这一设计选择，被后续几乎全部编码 agent（OpenHands/Confucius/KAT-Coder/Skywork-SWE）沿用。**这组最重要的一篇反直觉论文是 Agentless（2407.01489）**：证明在特定条件下，**去掉 agent 式的多轮交互循环、改用固定流水线反而表现更强**——是对"agent 循环是不是总比流水线好"这一默认假设的直接挑战，建议与 D 组 Less-Context-Better-Agents 对照读，两篇合起来构成全库"更复杂的设计不一定更好"这条批判性主线的两个支点。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| OpenHands Software Agent SDK (2511.03690) ⭐ | [报告](2511.03690-openhands-software-agent-sdk.md) | 开源编码 agent 旗舰 SDK | 前沿 2025 |
| Confucius Code Agent (2512.10398) | [报告](2512.10398-confucius-code-agent.md) | 前沿编码 agent 系统 | 前沿 2025 |
| KAT-Coder Technical Report (2510.18779) | [报告](2510.18779-kat-coder-technical-report.md) | 工业级编码 agent 技术报告 | 前沿 2025 |
| Skywork-SWE (2506.19290) | [报告](2506.19290-skywork-swe.md) | 面向 SWE-bench 优化的编码 agent | 前沿 2025 |
| CodeAct: Executable Code Actions (2402.01030) ⭐canon | [报告](2402.01030-codeact-executable-code-actions.md) | 可执行代码统一动作空间，**E 组 canon 支点** | canon 2024 |
| Agentless（无 agent 反而更强）(2407.01489) | [报告](2407.01489-agentless.md) | **重要反方**：固定流水线胜过 agent 循环 | 2024 |
| SWE-Fixer (2501.05040) | [报告](2501.05040-swe-fixer.md) | 检索+修复两阶段编码系统 | 前沿 2025 |
| MASAI: Modular Architecture for SWE (2406.11638) | [报告](2406.11638-masai-modular-architecture-swe.md) | 模块化多 agent 架构解 SWE 任务 | 2024 |
| DARS: Dynamic Action Resampling (2503.14269) | [报告](2503.14269-dars-dynamic-action-resampling.md) | 动态动作重采样提升成功率 | 前沿 2025 |
| AutoCodeRover (2404.05427) | [报告](2404.05427-autocoderover.md) | 结合代码结构分析的自动化修复 | 2024 |

---

## F. Web / 计算机使用 / GUI Agent（7 篇，~39.0 万字符）——E/T 层

**这组回答**：Agent 怎么在真实/仿真的数字环境（网页、桌面操作系统）里执行任务，纯文本表征够不够，要不要上视觉。
**情况**：Mind2Web 定义了"开放式任务+三级泛化切分"的评测协议 canon；WebArena 把评测从离线快照搬到可交互真实环境；OSWorld 进一步扩展到完整操作系统层面；VisualWebArena/WebVoyager/UI-TARS 构成一条清晰的"文本→视觉必需→纯视觉端到端"技术演进链，恰好逐条回应 Mind2Web 自陈的"未使用多模态信息"这一局限。**AgentSwing（2603.27490）是本库时间坐标最新的论文之一**（2026-03-31），同时也是与 D 组的姊妹篇——它把 D 组"该用哪种上下文管理策略"的路线之争，转化为"动态路由所有候选策略"这一整合性方案。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| WebArena (2307.13854) ⭐canon | [报告](2307.13854-webarena.md) | 可复现自托管网页环境+执行式评测 | canon 2023 |
| VisualWebArena (2401.13649) | [报告](2401.13649-visualwebarena.md) | WebArena 的视觉 grounding 扩展 | 2024 |
| OSWorld (2404.07972) ⭐canon | [报告](2404.07972-osworld.md) | 真实操作系统层面计算机使用基准 | canon 2024 |
| UI-TARS (2501.12326) | [报告](2501.12326-ui-tars.md) | 纯视觉端到端原生 GUI agent 模型 | 前沿 2025 |
| WebVoyager (2401.13919) | [报告](2401.13919-webvoyager.md) | 纯截图输入的视觉网页导航 agent | 2024 |
| Mind2Web (2306.06070) ⭐canon | [报告](2306.06070-mind2web.md) | **F 组协议起点**，137 网站开放式任务 | canon 2023 |
| AgentSwing: Parallel Context Routing (2603.27490) | [报告](2603.27490-agentswing-parallel-context-routing.md) | 上下文管理动态路由，**D 组姊妹篇**，**全库最新** | 前沿 2026-03 |

---

## G. Harness 评测 / Scaffold-Aware Eval（9 篇，~59.4 万字符）——O/V 层

**这组回答**：怎么设计出能公允测出"harness 到底行不行"的评测协议，而不是只测模型本身。
**情况**：**这是全库论点的实证收口组**——标杆 Harness-Bench（2605.27922，亲写）直接量化"同一模型换 harness 造成 23.8 分波动"，是"Agent = Model + Harness"这句话最硬的一份证据。SWE-bench/AgentBench/GAIA 三篇 canon 分别定义了"真实代码修复""跨环境综合能力""人类易AI难通用助手"三套此前最重要的评测协议；τ-bench 补上"人机协作+规则遵循"这一此前被忽视的维度；SWE-bench-CL/SWE-Evo/AgencyBench 代表把既有协议往"持续学习""长程演化""超长上下文"三个方向做增量拓展——**其中 SWE-bench-CL 意外贡献了一条比自身主题更有价值的元层教训**：直接套用静态基准的评测 harness 去跑一个结构不同的衍生数据集，会导致系统性、有时静默的评测失败。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| Harness-Bench (2605.27922) ⭐⭐ | [报告](2605.27922-harness-bench-measuring-harness-effects.md) | **库论点实证锚点**：harness 差异造成 23.8 分波动 | 前沿 2026·标杆 |
| SWE-bench (2310.06770) ⭐canon | [报告](2310.06770-swe-bench-resolve-github-issues.md) | 真实 GitHub issue 解决基准 | canon 2023 |
| Terminal-Bench: CLI Agents (2601.11868) | [报告](2601.11868-terminal-bench-cli-agents.md) | 终端/命令行 agent 评测基准 | 前沿 2026 |
| AgentBench (2308.03688) ⭐canon | [报告](2308.03688-agentbench-evaluating-llms-as-agents.md) | 跨 8 类环境评测 LLM-as-agent | canon 2023 |
| τ-bench: Tool-Agent-User (2406.12045) | [报告](2406.12045-tau-bench-tool-agent-user.md) | 客服人机协作+规则遵循，pass^k 新指标 | 2024 |
| GAIA (2311.12983) ⭐canon | [报告](2311.12983-gaia-general-ai-assistants.md) | "对人类简单、对 AI 难"通用助手基准 | canon 2023 |
| SWE-bench-CL: Continual Learning (2507.00014) | [报告](2507.00014-swe-bench-cl-continual-learning.md) | 持续学习维度，**意外贡献评测方法论元层教训** | 前沿 2025·预印本 |
| SWE-Evo: Long-Horizon Evolution (2512.18470) | [报告](2512.18470-swe-evo-long-horizon-software-evolution.md) | 长程软件演化场景评测 | 前沿 2025 |
| AgencyBench: 1M-Token Autonomous (2601.11044) | [报告](2601.11044-agencybench-1m-token-autonomous-agents.md) | 百万 token 级超长跨度自主性评测 | 前沿 2026 |

---

## H. 可靠性 / 安全 / 可观测 / 沙箱（6 篇，~29.5 万字符）——V 层

**这组回答**：Harness 出了问题怎么办——怎么提前防、怎么当场测、怎么系统性理解、怎么事后查、怎么恢复。
**情况**：全库体量最小但**内部结构最完整**的一组，六篇恰好拼出一条完整链条：**LlamaFirewall（预防）→ AgentDojo（检测评估，H 组事实标准基准，被本组另两篇论文直接复用）→ Systems-Security-Foundations（理论化系统性梳理，SoK 元层框架）→ AgentRacer（归因，定位具体是哪一步出的错）→ Hell-or-High-Water（恢复，知道错在哪之后能不能想出新方案）→ Fault-Tolerant Sandboxing（贯穿全流程的执行环境隔离）**。建议读完全组后，用 Systems-Security-Foundations 提出的四条经典安全原则（最小权限/完全中介/TCB 完整性/安全信息流）回头重新审视其余 5 篇各自的具体贡献。

| 论文 (arXiv) | 报告 | 一句话情况 | 坐标 |
|---|---|---|---|
| LlamaFirewall (2505.03574) | [报告](2505.03574-llamafirewall-guardrail-system.md) | Meta 生产环境三层防御护栏，**预防** | 前沿 2025 |
| AgentDojo (2406.13352) | [报告](2406.13352-agentdojo-prompt-injection-eval.md) | 提示注入攻防动态框架，**检测评估**，H 组事实标准 | 前沿 2024 |
| Fault-Tolerant Sandboxing for Coding Agents (2512.12806) | [报告](2512.12806-fault-tolerant-sandboxing-coding-agents.md) | 容错沙箱系统架构，**隔离执行** | 前沿 2025·早期原型 |
| Systems Security Foundations (2512.01295) | [报告](2512.01295-systems-security-foundations-agentic-computing.md) | 经典系统安全学审视 agent 安全的 SoK，**元层框架** | 前沿 2025-2026 |
| Hell or High Water (2508.11027) | [报告](2508.11027-hell-or-high-water-agentic-recovery.md) | 外部故障后备用方案规划能力评测，**恢复**，H 组收官 | 前沿 2025·COLM |
| AgentRacer (2509.03312) | [报告](2509.03312-agentracer-failure-attribution.md) | Agent 失败根因定位方法，**归因** | 前沿 2025 |

---

> **74/74 完成**。配套阅读见姊妹库 [`auto-research-frontier`](../../auto-research-frontier/paper-reports/CATALOG-by-type.md)（另 74 篇，"会做科研"方向）。
> 一句话收口（贯穿全 74 篇）：**Agent = Model + Harness**——控制循环（B）决定"每步想什么做什么"，工具接口（C）决定"能不能正确使唤外部能力"，
> 上下文管理（D）决定"长程运行会不会忘事/累积漂移"，评测协议（G）决定"我们说的'更好'是不是真的更好"，可靠性护栏（H）决定"出问题了能不能兜住"——
> 同一个模型，换一套 harness，Harness-Bench 实测能差出 **23.8 分**；而 D 组、E 组内部反复出现的"更复杂设计 vs 更简单方案哪个更好"的路线之争
> （Less-Context-Better-Agents vs 主流记忆系统、Agentless vs agent 循环），提醒我们：**harness 每加一层复杂度，都需要用实证去检验它是否真的物有所值**。
