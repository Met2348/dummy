# 组会汇报文档库 · agent-harness 74 篇精读（PPT 风格 · 50 分钟级）

> 为 [`../papers/`](../papers/) 里**全部 74 篇** agent-harness 论文各写的一份**约 20 页、50 分钟组会汇报级**中文文档。
> 每篇都**真读了 PDF**：intention/why > how，先直觉后定义、数字标 §/Table/Eq 出处，诚实区分「论文宣称 vs 批判/局限」，
> 缺失一律标"原文未给出"、不编造。**贯穿全库的核心论点**：**Agent = Model + Harness**——能力/可信度有一大半压在
> harness（控制循环/工具接口/上下文管理/评测协议/可靠性护栏）上，而不只是模型本身。
>
> 是继 [`auto-research-frontier`](../../auto-research-frontier/paper-reports/README.md)（"会做科研"）之后的**第二个大规模文献库**，
> 聚焦"把 LLM 变成能干活的 agent 的底座"这条新方向。规范三层：v1 硬规范 + v2（Why 三连 + Inspires-Us）+ 本库
> [`_STYLE-GUIDE-harness.md`](_STYLE-GUIDE-harness.md)（Θ1–Θ5 harness 专属：E/T/C/L/O/V 六层分类、回扣
> Agent=Model+Harness、Inspires-Us 强制打到我们自己的 harness、canon/前沿时间坐标诚实标注、"harness>model"不绝对化）。
> v2 标杆：[`2605.27922 Harness-Bench`](2605.27922-harness-bench-measuring-harness-effects.md)（库论点实证锚点，亲写）。
> 建设全程见 [`PROGRESS.md`](PROGRESS.md)。

## 怎么用

- **组会主讲**：每篇按约 20 页骨架组织，二级标题即"幻灯片"，配 `> 主讲提示`（该页该说什么）；文末有"延伸讨论问题""一页纸速记"，每篇都有强制的 **`## ★ 对我们的启发（Inspires Us）`** 一节，直接给"这对我们自己 harness 意味着什么、下一步能试什么"。
- **建议顺序**：先读 A 组 [2603.25723](2603.25723-natural-language-agent-harnesses.md)（坐标系）→ B 组 [2210.03629 ReAct](2210.03629-react-reasoning-and-acting.md)（控制循环祖先）→ 按兴趣进 C/D/E/F（能力构建）→ G 组 [2605.27922 Harness-Bench](2605.27922-harness-bench-measuring-harness-effects.md)（论点实证）→ H 组（可靠性/安全收口）。
- **六层分类（E/T/C/L/O/V）**：Environment 环境 / Tools 工具 / Context 上下文 / Loop 控制循环 / Observability 可观测 / Validation 校验——每篇标题行标注其主打层，可按层跨组检索。
- **配套阅读**：姊妹库 [`auto-research-frontier`](../../auto-research-frontier/paper-reports/README.md)（"会做科研"），两库合起来覆盖"agent 的两条新方向"。

---

## A. 综述 / 框架与定义（8 篇）——先读，建坐标系

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| Natural Language Agent Harnesses (2603.25723) | [报告](2603.25723-natural-language-agent-harnesses.md) | "Harness"这个词在 agent 语境下的定义与分类法 |
| Recreate: Experience-Driven Domain Agents (2601.11100) | [报告](2601.11100-recreate-experience-driven-domain-agents.md) | 从经验积累构建领域专属 agent 的框架 |
| Inside the Scaffold: Coding Agent Taxonomy (2604.03515) | [报告](2604.03515-inside-the-scaffold-coding-agent-taxonomy.md) | 编码 agent 脚手架的系统分类学 |
| AI Agent Systems: Architectures/Applications/Evaluation (2601.01743) | [报告](2601.01743-ai-agent-systems-architectures-applications-evaluation.md) | 架构-应用-评测三位一体综述 |
| Survey of Context Engineering for LLMs (2507.13334) | [报告](2507.13334-survey-of-context-engineering-for-llms.md) | 上下文工程系统综述，D 组的理论前置 |
| Memory in the Age of AI Agents (2512.13564) | [报告](2512.13564-memory-in-the-age-of-ai-agents.md) | Agent 记忆机制的最新系统综述 |
| Survey on LLM-based Autonomous Agents (2308.11432) | [报告](2308.11432-survey-on-llm-based-autonomous-agents.md) | 早期奠基性综述，agent 组件划分的公认参照 |
| Externalization in LLM Agents: A Review (2604.08224) | [报告](2604.08224-externalization-in-llm-agents-review.md) | "把认知过程外化"这一设计原则的综述 |

## B. 控制循环 / 推理-行动范式（10 篇，含 5 篇 canon 脊柱）——L 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐ ReAct (2210.03629) | [报告](2210.03629-react-reasoning-and-acting.md) | 推理+行动交错的 agent 控制循环**祖先** |
| ⭐ Reflexion (2303.11366) | [报告](2303.11366-reflexion-verbal-reinforcement-learning.md) | 用语言反思代替梯度更新的自我改进 |
| ⭐ Tree of Thoughts (2305.10601) | [报告](2305.10601-tree-of-thoughts.md) | 把推理过程建模为可搜索的树 |
| Language Agent Tree Search / LATS (2310.04406) | [报告](2310.04406-language-agent-tree-search-lats.md) | ToT + MCTS + 真实环境反馈的融合 |
| Plan-and-Solve Prompting (2305.04091) | [报告](2305.04091-plan-and-solve-prompting.md) | 先规划后执行的两阶段提示范式 |
| ⭐ Inner Monologue (2207.05608) | [报告](2207.05608-inner-monologue-embodied-planning.md) | 具身智能体的内心独白式规划 |
| ⭐ Self-Refine (2303.17651) | [报告](2303.17651-self-refine-iterative-self-feedback.md) | 无需训练的迭代自我反馈精修 |
| General Modular Harness / Gaming Agents (2507.11633) | [报告](2507.11633-general-modular-harness-gaming-agents.md) | 通用模块化 harness 在游戏 agent 上的验证 |
| From Agent Loops to Structured Graphs (2604.11378) | [报告](2604.11378-from-agent-loops-to-structured-graphs.md) | 控制循环从线性到图结构的演化 |
| FlowSteer: Reinforced Workflow Orchestration (2602.01664) | [报告](2602.01664-flowsteer-reinforced-workflow-orchestration.md) | 强化学习编排工作流的控制策略 |

## C. 工具接口 / Agent-Computer Interface（8 篇）——T 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐ SWE-agent: ACI (2405.15793) | [报告](2405.15793-swe-agent-agent-computer-interface.md) | Agent-Computer Interface **概念锚点** |
| ⭐ Toolformer (2302.04761) | [报告](2302.04761-toolformer-self-taught-tool-use.md) | LLM 自学何时/如何调用工具 |
| ToolLLM / ToolBench (2307.16789) | [报告](2307.16789-toolllm-toolbench-16000-apis.md) | 16000+ 真实 API 的大规模工具学习 |
| Gorilla: LLM Connected with Massive APIs (2305.15334) | [报告](2305.15334-gorilla-llm-connected-massive-apis.md) | 检索增强的海量 API 调用 |
| ToolACE: Winning Function Calling (2409.00920) | [报告](2409.00920-toolace-winning-function-calling.md) | 高质量合成数据驱动的函数调用训练 |
| Less is More: Function Calling at the Edge (2411.15399) | [报告](2411.15399-less-is-more-function-calling-edge.md) | 工具裁剪提升准确率，**Vercel 现象学术版** |
| MemTool: Short-Term Memory for Tool Calling (2507.21428) | [报告](2507.21428-memtool-short-term-memory-tool-calling.md) | 工具调用场景的短期记忆机制 |
| FuncBenchGen: Contamination-Free Eval (2509.26553) | [报告](2509.26553-funcbenchgen-contamination-free-eval.md) | 防止数据污染的函数调用评测生成 |

## D. 上下文工程 / 记忆（16 篇，最大组）——C 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐ MemGPT: LLMs as Operating Systems (2310.08560) | [报告](2310.08560-memgpt-llms-as-operating-systems.md) | 把 OS 虚拟内存分页思想搬进 LLM 记忆管理 |
| MemoryBank (2305.10250) | [报告](2305.10250-memorybank-long-term-memory.md) | 长期记忆 + 遗忘曲线机制 |
| Mem0: Production Long-Term Memory (2504.19413) | [报告](2504.19413-mem0-production-long-term-memory.md) | 生产级长期记忆系统 |
| MEM1: Synergize Memory & Reasoning (2506.15841) | [报告](2506.15841-mem1-synergize-memory-reasoning.md) | 记忆与推理协同训练 |
| Mem-α: RL Memory Construction (2509.25911) | [报告](2509.25911-mem-alpha-rl-memory-construction.md) | 用强化学习学习如何构建记忆 |
| A-MEM: Agentic Memory / Zettelkasten (2502.12110) | [报告](2502.12110-a-mem-agentic-memory-zettelkasten.md) | 卡片盒笔记法启发的 agent 记忆，**与我们 MEMORY.md 的 `[[link]]` 同源** |
| MemAgent: Multi-Conv RL Memory (2507.02259) | [报告](2507.02259-memagent-multiconv-rl-memory.md) | 跨多轮对话的强化学习记忆管理 |
| IterResearch: Interaction Scaling (2511.07327) | [报告](2511.07327-iterresearch-interaction-scaling.md) | 迭代式交互规模化的记忆策略 |
| AgentFold: Proactive Context Folding (2510.24699) | [报告](2510.24699-agentfold-proactive-context-folding.md) | 主动折叠上下文而非被动截断 |
| ACON: Context Compression for Agents (2510.00615) | [报告](2510.00615-acon-context-compression-agents.md) | 面向 agent 场景的上下文压缩 |
| Memory as Action: Context Curation (2510.12635) | [报告](2510.12635-memory-as-action-context-curation.md) | 把"管理记忆"本身建模为一种动作 |
| MemSearcher: Reason-Search-Manage Memory (2511.02805) | [报告](2511.02805-memsearcher-reason-search-manage-memory.md) | 推理-搜索-记忆管理一体化 |
| AgentOCR: Optical Self-Compression (2601.04786) | [报告](2601.04786-agentocr-optical-self-compression.md) | 用"视觉渲染再 OCR"实现极限上下文压缩 |
| Agentic Memory: Unified LTM+STM (2601.01885) | [报告](2601.01885-agentic-memory-unified-ltm-stm.md) | 长短期记忆的统一框架 |
| RE-TRAC: Recursive Trajectory Compression (2602.02486) | [报告](2602.02486-re-trac-recursive-trajectory-compression.md) | 递归式轨迹压缩 |
| Less Context, Better Agents (2606.10209) | [报告](2606.10209-less-context-better-agents.md) | "少即是多"**反方压舱石**，D 组的批判性收尾 |

## E. 编码 / SWE Agent 集成系统（10 篇）——E/T 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐ OpenHands Software Agent SDK (2511.03690) | [报告](2511.03690-openhands-software-agent-sdk.md) | 开源编码 agent 旗舰 SDK |
| Confucius Code Agent (2512.10398) | [报告](2512.10398-confucius-code-agent.md) | 前沿编码 agent 系统 |
| KAT-Coder Technical Report (2510.18779) | [报告](2510.18779-kat-coder-technical-report.md) | 工业级编码 agent 技术报告 |
| Skywork-SWE (2506.19290) | [报告](2506.19290-skywork-swe.md) | 面向 SWE-bench 优化的编码 agent |
| ⭐ CodeAct: Executable Code Actions (2402.01030) | [报告](2402.01030-codeact-executable-code-actions.md) | 用可执行代码统一动作空间的 canon 设计 |
| Agentless（无 agent 反而更强）(2407.01489) | [报告](2407.01489-agentless.md) | **重要对照**：不用 agent 循环、纯流水线反而更强 |
| SWE-Fixer (2501.05040) | [报告](2501.05040-swe-fixer.md) | 检索+修复两阶段编码系统 |
| MASAI: Modular Architecture for SWE (2406.11638) | [报告](2406.11638-masai-modular-architecture-swe.md) | 模块化多 agent 架构解 SWE 任务 |
| DARS: Dynamic Action Resampling (2503.14269) | [报告](2503.14269-dars-dynamic-action-resampling.md) | 动态动作重采样提升编码成功率 |
| AutoCodeRover (2404.05427) | [报告](2404.05427-autocoderover.md) | 结合代码结构分析的自动化 bug 修复 |

## F. Web / 计算机使用 / GUI Agent（7 篇）——E/T 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐ WebArena (2307.13854) | [报告](2307.13854-webarena.md) | 可复现自托管网页环境 + 执行式评测的 canon |
| VisualWebArena (2401.13649) | [报告](2401.13649-visualwebarena.md) | WebArena 的视觉 grounding 扩展 |
| ⭐ OSWorld (2404.07972) | [报告](2404.07972-osworld.md) | 真实操作系统层面的计算机使用基准 |
| UI-TARS (2501.12326) | [报告](2501.12326-ui-tars.md) | 纯视觉端到端原生 GUI agent 模型 |
| WebVoyager (2401.13919) | [报告](2401.13919-webvoyager.md) | 纯截图输入的视觉网页导航 agent |
| ⭐ Mind2Web (2306.06070) | [报告](2306.06070-mind2web.md) | 首个通才 web agent 数据集，137 网站 canon |
| AgentSwing: Parallel Context Routing (2603.27490) | [报告](2603.27490-agentswing-parallel-context-routing.md) | 并行上下文管理策略动态路由，**D 组姊妹篇**，2026 最新前沿 |

## G. Harness 评测 / Scaffold-Aware Eval（9 篇）——O/V 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| ⭐⭐ Harness-Bench (2605.27922) | [报告](2605.27922-harness-bench-measuring-harness-effects.md) | **库论点实证锚点**：量化 harness 造成的性能波动，v2 标杆 |
| ⭐ SWE-bench (2310.06770) | [报告](2310.06770-swe-bench-resolve-github-issues.md) | 真实 GitHub issue 解决基准，evaluation harness canon |
| Terminal-Bench: CLI Agents (2601.11868) | [报告](2601.11868-terminal-bench-cli-agents.md) | 终端/命令行 agent 评测基准 |
| ⭐ AgentBench (2308.03688) | [报告](2308.03688-agentbench-evaluating-llms-as-agents.md) | 跨 8 类环境评测 LLM-as-agent 的早期系统性基准 |
| τ-bench: Tool-Agent-User (2406.12045) | [报告](2406.12045-tau-bench-tool-agent-user.md) | 客服场景人机协作+规则遵循评测，pass^k 新指标 |
| ⭐ GAIA (2311.12983) | [报告](2311.12983-gaia-general-ai-assistants.md) | "对人类简单、对 AI 难"通用助手基准 |
| SWE-bench-CL: Continual Learning (2507.00014) | [报告](2507.00014-swe-bench-cl-continual-learning.md) | 给编码 agent 加持续学习维度 |
| SWE-Evo: Long-Horizon Software Evolution (2512.18470) | [报告](2512.18470-swe-evo-long-horizon-software-evolution.md) | 长程软件演化场景评测 |
| AgencyBench: 1M-Token Autonomous Agents (2601.11044) | [报告](2601.11044-agencybench-1m-token-autonomous-agents.md) | 百万 token 级超长跨度自主性评测 |

## H. 可靠性 / 安全 / 可观测 / 沙箱（6 篇）——V 层

| 论文 (arXiv) | 报告 | 一句话 |
|---|---|---|
| LlamaFirewall (2505.03574) | [报告](2505.03574-llamafirewall-guardrail-system.md) | Meta 生产环境三层防御护栏系统 |
| AgentDojo (2406.13352) | [报告](2406.13352-agentdojo-prompt-injection-eval.md) | 提示注入攻防的动态可扩展评测框架，H 组事实标准基准 |
| Fault-Tolerant Sandboxing for Coding Agents (2512.12806) | [报告](2512.12806-fault-tolerant-sandboxing-coding-agents.md) | 容错沙箱系统架构 |
| Systems Security Foundations for Agentic Computing (2512.01295) | [报告](2512.01295-systems-security-foundations-agentic-computing.md) | 用四十年系统安全学审视 agent 安全的 SoK，**H 组元层框架** |
| Hell or High Water: Agentic Recovery (2508.11027) | [报告](2508.11027-hell-or-high-water-agentic-recovery.md) | 外部故障后的备用方案规划能力评测，H 组收官 |
| AgentRacer: Failure Attribution (2509.03312) | [报告](2509.03312-agentracer-failure-attribution.md) | Agent 失败根因定位方法 |

---

> **74/74 完成**（A:8 + B:10 + C:8 + D:16 + E:10 + F:7 + G:9 + H:6）。配套阅读见姊妹库
> [`auto-research-frontier`](../../auto-research-frontier/paper-reports/README.md)（另 74 篇，"会做科研"方向）。
> 一句话收口（贯穿全 74 篇）：**Agent 的能力上限由模型决定，但能不能稳定、安全、高效地把这个上限兑现出来，几乎全压在 harness 上**——
> 控制循环设计（B）、工具接口（C）、上下文管理（D）、评测协议（G）、可靠性护栏（H）任何一层没做好，
> 同一个模型跑出来的表现能差出几十个百分点（Harness-Bench 实测 23.8 分）。
