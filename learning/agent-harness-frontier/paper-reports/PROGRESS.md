# Agent-Harness 文献报告库 · 建设账本（source of truth，抗 compaction）

> 任务：为 `papers/` 里**全部 74 篇** agent-harness 论文各写一份约 20 页、50 分钟组会级、PPT 风格中文 md。
> 老师新方向（2026-06-27）：「Coding 被 agent 接管 → 下一个是 research；**harness 是另一个新方向，也要大规模文献调研**」。
> 用户决定：**通用 harness 全景 · ≥70 篇 · 仅报告库 · 全复用工业化流水线**。
> 贯穿论点：**Agent = Model + Harness**——能力/可信度有一大半压在 harness 上（对应 auto-research 的「独立验证收口」）。

## 规范（三层）
- v1 硬规范 + v2（Why 三连 + Inspires-Us）+ 本库 [`_STYLE-GUIDE-harness.md`](_STYLE-GUIDE-harness.md)（Θ1–Θ5 harness 专属：E/T/C/L/O/V 分层、回扣 Agent=Model+Harness、Inspires-Us 打到自己 harness、canon/前沿坐标、regime 诚实）。
- 标杆：我亲写 1 篇 v2 harness 标杆（**Harness-Bench 2605.27922** 或 OpenHands SDK），其余派子代理并行对齐。

## 执行方式（同 auto-research，已验证抗中断）
- 子代理真读对应 PDF（`papers/<id>-<slug>.pdf`），严格按规范写到 `paper-reports/<id>-<slug>.md`，忠于原文、标章节/公式编号、不编造。
- **产物落盘 + 本账本做唯一真相**；分批 commit；每次 `git add` 用**显式路径**，绝不 `git add .`。
- 抽检：每批挑 1–2 篇核对（公式前直觉/先定义符号、Inspires-Us 落地、§/Table 出处、Why 三连齐）。

## 进度（74 篇）　状态：⬜ 未做 / 🟡 进行中 / ✅ 已写并抽检

### A. 综述 / 框架与定义（8）✅ 全组完成
- ✅ 2603.25723 natural-language-agent-harnesses (51k)
- ✅ 2601.11100 recreate-experience-driven-domain-agents (46k)
- ✅ 2604.03515 inside-the-scaffold-coding-agent-taxonomy (84k)
- ✅ 2601.01743 ai-agent-systems-architectures-applications-evaluation (46k)
- ✅ 2507.13334 survey-of-context-engineering-for-llms (46k)
- ✅ 2512.13564 memory-in-the-age-of-ai-agents (50k)
- ✅ 2308.11432 survey-on-llm-based-autonomous-agents (50k)
- ✅ 2604.08224 externalization-in-llm-agents-review (76k)

### B. 控制循环 / 推理-行动范式（10，含 5 篇 canon 脊柱）✅ 全组完成
- ✅ 2210.03629 react-reasoning-and-acting ⭐canon (42k)
- ✅ 2303.11366 reflexion-verbal-reinforcement-learning ⭐canon (40k)
- ✅ 2305.10601 tree-of-thoughts ⭐canon (41k)
- ✅ 2310.04406 language-agent-tree-search-lats (42k)
- ✅ 2305.04091 plan-and-solve-prompting (35k)
- ✅ 2207.05608 inner-monologue-embodied-planning ⭐canon (38k)
- ✅ 2303.17651 self-refine-iterative-self-feedback ⭐canon (42k)
- ✅ 2507.11633 general-modular-harness-gaming-agents (49k)
- ✅ 2604.11378 from-agent-loops-to-structured-graphs (53k)
- ✅ 2602.01664 flowsteer-reinforced-workflow-orchestration (53k)

### C. 工具接口 / Agent-Computer Interface（8）✅ 全组完成
- ✅ 2405.15793 swe-agent-agent-computer-interface ⭐canon(ACI 锚点) (66k)
- ✅ 2302.04761 toolformer-self-taught-tool-use ⭐canon (67k)
- ✅ 2307.16789 toolllm-toolbench-16000-apis (64k)
- ✅ 2305.15334 gorilla-llm-connected-massive-apis (55k)
- ✅ 2409.00920 toolace-winning-function-calling (51k)
- ✅ 2411.15399 less-is-more-function-calling-edge (63k·Vercel现象学术版)
- ✅ 2507.21428 memtool-short-term-memory-tool-calling (45k)
- ✅ 2509.26553 funcbenchgen-contamination-free-eval (68k)

### D. 上下文工程 / 记忆（16，最大组）　4a✅ / 4b⬜
- ✅ 2310.08560 memgpt-llms-as-operating-systems ⭐canon (51k)
- ✅ 2305.10250 memorybank-long-term-memory (47k)
- ✅ 2504.19413 mem0-production-long-term-memory (49k)
- ✅ 2506.15841 mem1-synergize-memory-reasoning (57k)
- ✅ 2509.25911 mem-alpha-rl-memory-construction (54k)
- ✅ 2502.12110 a-mem-agentic-memory-zettelkasten (48k·我们MEMORY.md的[[link]]同源)
- ✅ 2507.02259 memagent-multiconv-rl-memory (49k)
- ✅ 2511.07327 iterresearch-interaction-scaling (49k)
- ✅ 2510.24699 agentfold-proactive-context-folding (56k)
- ✅ 2510.00615 acon-context-compression-agents (63k)
- ✅ 2510.12635 memory-as-action-context-curation (65k)
- ✅ 2511.02805 memsearcher-reason-search-manage-memory (64k)
- ✅ 2601.04786 agentocr-optical-self-compression (64k)
- ✅ 2601.01885 agentic-memory-unified-ltm-stm (59k)
- ✅ 2602.02486 re-trac-recursive-trajectory-compression (63k)
- ✅ 2606.10209 less-context-better-agents (69k·"少即是多"反方压舱石) ✅ D组16篇全完成

### E. 编码 / SWE Agent 集成系统（10）✅ 全组完成
- ✅ 2511.03690 openhands-software-agent-sdk ⭐ (67k)
- ✅ 2512.10398 confucius-code-agent (67k)
- ✅ 2510.18779 kat-coder-technical-report (68k)
- ✅ 2506.19290 skywork-swe (60k)
- ✅ 2402.01030 codeact-executable-code-actions ⭐canon (53k)
- ✅ 2407.01489 agentless（无 agent 反而更强，重要对照·68k）
- ✅ 2501.05040 swe-fixer (64k)
- ✅ 2406.11638 masai-modular-architecture-swe (65k)
- ✅ 2503.14269 dars-dynamic-action-resampling (62k)
- ✅ 2404.05427 autocoderover (51k)

### F. Web / 计算机使用 / GUI Agent（7）✅ 全组完成（4篇主线程直接撰写，应对子代理日限额中断）
- ✅ 2307.13854 webarena ⭐canon (76k)
- ✅ 2401.13649 visualwebarena (36.2k · 主线程直接撰写)
- ✅ 2404.07972 osworld ⭐canon (70k)
- ✅ 2501.12326 ui-tars (38.0k · 主线程直接撰写)
- ✅ 2401.13919 webvoyager (64k)
- ✅ 2306.06070 mind2web ⭐canon (69.7k · 主线程续写：子代理撞限额于§13截断，续写§14–20+Inspires-Us)
- ✅ 2603.27490 agentswing-parallel-context-routing (36.5k · 主线程直接撰写，2026-03-31前沿，D组姊妹篇)

### G. Harness 评测 / scaffold-aware eval（9）✅ 全组完成
- ✅ 2605.27922 harness-bench-measuring-harness-effects ⭐⭐(库论点实证锚点 · **我亲写 v2 标杆**)
- ✅ 2310.06770 swe-bench-resolve-github-issues ⭐canon (84.4k · 子代理配额已恢复，验证完整流程正常)
- ✅ 2601.11868 terminal-bench-cli-agents (88.9k · 子代理，13篇批次中撞5h限额但已落盘)
- ✅ 2308.03688 agentbench-evaluating-llms-as-agents (91.8k · 子代理自报failed但磁盘已落盘，协议再次验证)
- ✅ 2406.12045 tau-bench-tool-agent-user (34.9k · 主线程直接撰写)
- ✅ 2311.12983 gaia-general-ai-assistants (88.5k · 子代理自报failed但磁盘已落盘)
- ✅ 2507.00014 swe-bench-cl-continual-learning (34.5k · 主线程直接撰写，诚实标注论文为预印本/审稿中性质，核心实证部分"进行中"未编造)
- ✅ 2512.18470 swe-evo-long-horizon-software-evolution (79.9k)
- ✅ 2601.11044 agencybench-1m-token-autonomous-agents (64.8k)

### H. 可靠性 / 安全 / 可观测 / 沙箱（6）
- ⬜ 2505.03574 llamafirewall-guardrail-system （撞5h限额未落盘，待主线程撰写）
- ⬜ 2406.13352 agentdojo-prompt-injection-eval （唯一真正未产出内容的一篇，撞5h限额，待主线程撰写）
- ✅ 2512.12806 fault-tolerant-sandboxing-coding-agents (75.5k · 13篇并行批次中唯一在进程中断前完整落盘的一篇，结构自检通过)
- ⬜ 2512.01295 systems-security-foundations-agentic-computing （撞5h限额未落盘，待主线程撰写）
- ⬜ 2508.11027 hell-or-high-water-agentic-recovery （撞5h限额未落盘，待主线程撰写）
- ✅ 2509.03312 agentracer-failure-attribution (72.1k · 子代理自报failed但磁盘已落盘)
- ⬜ 2509.03312 agentracer-failure-attribution

## 已提交批次（commit + 含哪几篇）
- ✅ 基建 b2ae2a9：下载脚本(74篇)+harness规范Θ1–Θ5+账本+spec；**74/74 PDF 全部下成**（零失败、零臆造 ID）
- ✅ 标杆 (待提交)：2605.27922 harness-bench（我亲写，25.9k 字，演示 Θ1–Θ5 + Why 三连 + Inspires-Us）
- ✅ 批次 1（A 组 8 篇）：2603.25723/2601.11100/2604.03515/2601.01743/2507.13334/2512.13564/2308.11432/2604.08224（2 篇撞 429 但已落盘）
- ✅ 批次 2（B 组 10 篇·全 L 层，含 5 canon：ReAct/Reflexion/ToT/Inner-Monologue/Self-Refine + LATS/Plan-Solve/Modular-Harness/Graph-Harness/FlowSteer）
- ✅ 批次 3（C 组 8 篇·T 层·工具/ACI：SWE-agent/Toolformer/ToolLLM/Gorilla/ToolACE/Less-is-More/MemTool/FuncBenchGen）
- ✅ 批次 4a（D 组前 8·C 层：MemGPT canon/MemoryBank/Mem0/MEM1/Mem-α/A-MEM/MemAgent/IterResearch）※ 8 篇均撞 5h 限额但已落盘
- ✅ 批次 4b（D 组后 8·C 层：AgentFold/ACON/Memory-as-Action/MemSearcher/AgentOCR/Agentic-Memory/RE-TRAC/Less-Context）→ **D 组 16 篇全完成**
- ✅ 批次 5（E 组编码 10：OpenHands SDK 旗舰 + CodeAct canon + Agentless反方 + MASAI/Confucius/KAT/Skywork/SWE-Fixer/DARS/AutoCodeRover）
- ✅ 修复 (已提交)：统一3篇启发节标题格式（去除§N前缀，全库可检索一致性审计）
- ✅ 批次 6（F 组 Web/GUI 7 篇全完成）：webarena/osworld/webvoyager 子代理时期完成；**mind2web 子代理连续2次撞"api key 日限额已用完"**后，改为主线程直接用 pdftotext 提取PDF文本撰写，mind2web/visualwebarena/ui-tars/agentswing 4 篇均由主线程亲自完成（分别69.7k/36.2k/38.0k/36.5k字，逐篇结构自检+独立commit）
- ⬜ 批次 7（G 组 9，标杆已完成，剩 8）／批次 8（H 组 6）——子代理派发路径持续不可用，继续全部改主线程直接撰写

## 备选/缓冲（已核验，下载失败可顶上）
- D 2601.02553 simplemem · D 2509.24704 memgen · F 2604.01664 contextbudget · G 2602.22769 ama-bench

## canon / 前沿配比
约 **30 篇 2022–2024 基石**（harness 必备脊柱，全顶会）+ 约 **44 篇 2025–2026 前沿**（60% 为新）。

## 安全护栏（不可破）
- `learning/agent-foundations/lectures/02-react.md` 全程**不碰、不暂存、不提交**（用户自己的笔记）。
- harness 跑子进程 `--json-out/--md-out` 必指 **temp**，绝不覆盖基线。
- PDF 走 `.gitignore`（已加 `learning/agent-harness-frontier/papers/*.pdf`），只提交 `.md`。
- **不 push，除非用户明说**；分支 `ERIC-3080Ti/paper-guides`。

## 与 [[auto-research-reading-focus]] 的关系
姊妹库：auto-research（74 篇，"会做科研"）+ agent-harness（74 篇，"把 LLM 变成能干活的 agent 的底座"）= 两条新方向。
