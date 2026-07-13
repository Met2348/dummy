# Core and Direct-Competitor Full-Text Dossier

This dossier covers **108** papers: all screened core papers plus every full-corpus item classified as a direct competitor.
Evidence is extracted from local full text and links back to a per-paper card and the source document.

## context-memory-tools

### [2604.20938: HARBOR: Automated Harness Optimization](../reading-cards/2604.20938.md)

- Class: `core-selected`; pages: 14; local text characters: 79458
- Full text: [PDF](../papers-all/2604.20938.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=19

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2607.04528: Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents](../reading-cards/2607.04528.md)

- Class: `core-selected`; pages: 28; local text characters: 112746
- Full text: [PDF](../papers-all/2607.04528.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=19; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] First, we introduce a belief-rollout elicit a K-step belief rollout (Hao et al., 2023; Yao diagnostic for controlled cross-harness compari- et al., 2023a; Deng et al., 2025), where each step son in software-agent evaluation.
- [introduction, p. 1] To isolate the effect of the we introduce BIWM, a no-training protocol that harness, we hold the task, environment, and base canonicalises beliefs, preserves blocked and re- LLM fixed, vary only the harness configuration, paired branches, records verification masks, ex- and compare the resulting bel...

**Result evidence**

- [experiments, p. 4] This formali- anisms genuinely improve safety, but they have an sation follows the agent-evaluation convention of underappreciated side effect: blocking an action separating the model, environment, and interaction also censors information.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.22953: Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents](../reading-cards/2606.22953.md)

- Class: `core-selected`; pages: 17; local text characters: 53630
- Full text: [PDF](../papers-all/2606.22953.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=17; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce replay pairing, a diagnostic that runs the same trajectory with and without the plan in history and measures hidden-state cosine distance.

**Result evidence**

- [experiments, p. 5] Pre-plan steps are a sanity check: A and B are identical before the guard, and we observe mean signal < 10−8 at steps −2, −1 (floating-point noise), confirming the replay pairing is correct.

**Limitations / caution evidence**

- [limitations, p. 9] The plan is a case study in a general question: which tokens an agent can safely drop.
- [limitations, p. 9] The same replay-pairing test applies to other tokens that may live only in context, such as safety instructions, constraints, and tool schemas.
- [limitations, p. 9] Reasoning models add a second mechanism: their <think> traces re-derive plan content step by step, partially substituting for keeping the plan in the window (and why naive replay pairing under-counts on them).

### [2604.08224: Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering](../reading-cards/2604.08224.md)

- Class: `core-selected`; pages: 54; local text characters: 219487
- Full text: [PDF](../papers-all/2604.08224.pdf)
- Five-idea scores: PREQ-Harness=12; Harness Transport=7; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [conclusion, p. 44] This paper has argued that externalization is the transition logic connecting many of the most important developments in LLM agents.
- [conclusion, p. 44] Reliable agency increasingly depends on relocating selected cognitive burdens out of the model and into explicit infrastructure: memory externalizes state across time, skills externalize procedural expertise, protocols externalize interaction structure, and the harness coordinates these layers into...
- [conclusion, p. 44] From this perspective, the move from weights to context to harness is not just a sequence of engineering tricks.

### [2603.28052: Meta-Harness: End-to-End Optimization of Model Harnesses](../reading-cards/2603.28052.md)

- Class: `core-selected`; pages: 26; local text characters: 84550
- Full text: [PDF](../papers-all/2603.28052.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=10; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Meta-Harness, an outer-loop system that searches over harness code for LLM applications.

**Result evidence**

- [experiments, p. 5] Meta-Harness achieves a stronger Harness improves online text classification ac- accuracy-context Pareto frontier than all curacy while using a smaller input context. comparison methods. designing a harness for text classification.
- [experiments, p. 5] The results show a large gap in favor of the full interface: scores-only reaches 34.6 median and 41.3 best accuracy, while scores-plus-summary reaches 34.9 median and 38.7 best.
- [abstract, p. 1] On online text classification, Meta-Harness improves over a state-of-the-art context management system by 7.7 points while using 4× fewer context tokens.

**Limitations / caution evidence**

- [related_work, p. 3] At a high level, Meta-Harness brings ideas from the broader literature on credit assignment and meta-learning [40; 46; 3; 17; 44; 2] in a new regime enabled by recent advances in coding agents.
- [related_work, p. 3] Rather than updating model weights, the system assigns credit at the harness level: it uses experience from past rollouts to deliberately reason about which steps and components are responsible for failures, then rewrites the external code that governs future behavior.
- [related_work, p. 3] More specifically, the method lies at the intersection of several recent research threads; it is most directly related to work on adaptive access to external context, executable code search, and text optimization.

### [2605.29682: Scaling Laws for Agent Harnesses via Effective Feedback Compute](../reading-cards/2605.29682.md)

- Class: `core-selected`; pages: 29; local text characters: 111399
- Full text: [PDF](../papers-all/2605.29682.pdf)
- Five-idea scores: PREQ-Harness=3; Harness Transport=9; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Effective Feed- sions (Liu et al., 2025; Lee et al., 2026; Kim et al., back Compute (EFC), a trace-level scal- 2026a,b).
- [abstract, p. 1] We fur- nate for closed-loop agent harness performance? ther define Estimated-EFC, NRS-EFC, har- We propose Effective Feedback Compute (EFC) ness efficiency η, and task-demand normal- ization for realistic traces and heterogeneous as this coordinate.
- [introduction, p. 1] We introduce EFC-A DAPTER, τ = {(st , at , ot , ut )}H t=1 , (1) an EFC-aware companion layer for existing har- nesses.

**Result evidence**

- [conclusion, p. 10] Across improve loop to update harness policies from ob- controlled, real, and held-out evaluations, EFC- served failures.
- [conclusion, p. 10] RTV and PDR represent test-time based coordinates outperform raw-compute base- trajectory scaling: RTV recursively selects among lines and SAS, and EFC-A DAPTER improves the sampled rollouts through comparison voting, and accuracy–cost tradeoff of existing harnesses under PDR conditions fresh attemp...
- [abstract, p. 1] Across synthetic, real, held-out, and it is informative, valid, non-redundant, and retained, prospective evaluations, EFC-based coordi- thereby separating raw spending from feedback nates outperform raw-compute baselines and that can change future behavior.

**Limitations / caution evidence**

- [conclusion, p. 10] We evaluate EFC-A DAPTER as a matched- budget companion layer attached to four represen- This paper shows that agent harness scaling is bet- tative baselines, instantiating three lines of agent- ter explained by effective feedback than by raw test- harness work. mini-SWE-agent (Yang et al., 2024) ti...
- [conclusion, p. 10] We introduce Effective Feedback represents lightweight closed-loop coding agents Compute (EFC), with Estimated-EFC, NRS-EFC, that solve tasks through execution, observation, and harness efficiency, and task-demand normalization, iterative repair.
- [conclusion, p. 10] AHE (Lin et al., 2026) represents to measure valid, informative, non-redundant, and harness self-evolution, using an evaluate–analyze– retained feedback relative to task demand.

### [2605.22166: Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents](../reading-cards/2605.22166.md)

- Class: `core-selected`; pages: 18; local text characters: 79590
- Full text: [PDF](../papers-all/2605.22166.pdf)
- Five-idea scores: PREQ-Harness=9; Harness Transport=8; MRT-Harness=0; Harness-C=0; ActiveHarness=1

**Contribution evidence**

- [introduction, p. 1] Recur- deterministic LLM agents, framing agent improve- ring training-trajectory failures are mapped to fixed ment as evolving the reusable interface between a interventions for environment contracts, procedural frozen model and a rule-governed environment. (2) skills, action realization, and trajec...

**Result evidence**

- [experiments, p. 6] Gain Improved ALFWorld 41.1% 75.7% +84% 17/18 -Bench AgentBench WebShop 31.4% 44.0% +40% 18/18 60 Best Performance (%) AgentBench OS Pass@1 34.7% 41.2% +19% 18/18 Final Avg.
- [experiments, p. 6] 20 Base 1 2 3 4 5 Airline Retail Telecom OS Webshop DB Alfworld 0.80 Iteration Qwen3-4B +0.25 +0.15 +0.32 +0.18 +0.16 +0.21 +0.72 Qwen2.5-7B +0.22 +0.05 +0.20 +0.10 +0.14 +0.17 +0.22 Figure 6: Training set performance improves steadily as Llama3.1-8B +0.03 -0.01 +0.18 +0.02 +0.20 +0.32 +0.77 the num...
- [abstract, p. 1] We pro- pose L IFE -H ARNESS, a lifecycle-aware run- Figure 1: Adapting the runtime harness, not the model. time harness that improves frozen LLM agents L IFE -H ARNESS keeps LLM weights fixed and evolves without changing model weights or evalua- reusable interface interventions from training trajec...

**Limitations / caution evidence**

- [limitations, p. 9] Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, This work focuses on deterministic, rule-governed Akhil Mathur, Alan Schelten, Amy Yang, Angela agent environments where the tool interface, feed- Fan, and 1 others.
- [limitations, p. 9] The llama 3 herd of models. back rules, and evaluation criteria are relatively arXiv e-prints, pages arXiv–2407. stable.
- [limitations, p. 9] This setting is common in database manipu- Chrisantha Fernando, Dylan Banarse, Henryk lation, web shopping, and policy-guided business Michalewski, Simon Osindero, and Tim Rock- workflows, and it enables failures to be reproduced täschel.

### [2606.20683: From Question Answering to Task Completion: A Survey on Agent System and Harness Design](../reading-cards/2606.20683.md)

- Class: `core-selected`; pages: 29; local text characters: 188163
- Full text: [PDF](../papers-all/2606.20683.pdf)
- Five-idea scores: PREQ-Harness=5; Harness Transport=0; MRT-Harness=0; Harness-C=6; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [experiments, p. 15] Task success remains the primary outcome met- and continuous evaluation practices feed these signals back ric, but meaningful harness comparison requires a richer into system improvement. reading of results along several additional dimensions: Benchmarks as task specifications.

**Limitations / caution evidence**

- [related_work, p. 4] Two abstraction levels are often conflated in the agent model is necessary because it provides the general-purpose literature.
- [related_work, p. 4] At the functional level, an agent is a goal-directed cognitive capabilities that make open-ended task comple- closed-loop system: it perceives an external environment, tion possible.
- [related_work, p. 4] It is not sufficient, however, because the model maintains task state, reasons and decides, executes actions, does not by itself define what observations are available, and adapts from feedback.

### [2605.05716: More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding](../reading-cards/2605.05716.md)

- Class: `core-selected`; pages: 12; local text characters: 45686
- Full text: [PDF](../papers-all/2605.05716.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=5; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present evidence that this assumption is often To characterize the interaction structure, we wrong for prompt-based scaffolding, and wrong in fit a main-effects regression (R2 =0.916, ad- a specific, measurable, and nuanced way. justed R2 =0.899, LOOCV R2 =0.872) that de- Consider Llama-3.1-8B on...

**Result evidence**

- [results, p. 6] The robust evidence against submodu- larity comes from the marginal-ratio distribution: This achieves R2 =0.937, but adjusted R2 drops γmed =0.52 with CI well below 1. to 0.878 and LOOCV R2 drops to 0.748.
- [results, p. 6] We characterize these structurally rather than from individual component effects alone, without testing each triple individually; none of the top-20 modeling interaction terms—practitioners can as- violations is individually significant after Bonfer- sess component value independently.
- [results, p. 6] T +SR+R achieves 0.430—a 95% improvement Table 7: Length-matched control (3-seed mean).

**Limitations / caution evidence**

- [limitations, p. 8] all single-component marginals are negative— missing the true optimum T+SR+R (accuracy Model coverage.
- [limitations, p. 8] We validate across two open- 0.430, a 95% improvement). weight families (Llama, Qwen) with models ≤70B, plus one closed-source API model (Claude Haiku benchmarks.
- [limitations, p. 8] CCI may differ for larger frontier models lects no human data, and poses no foreseeable dual- (GPT-4, Claude Opus) with stronger instruction- use risks.

### [2606.14674: AgentSpec: Understanding Embodied Agent Scaffolds Through Controlled Composition](../reading-cards/2606.14674.md)

- Class: `core-selected`; pages: 40; local text characters: 159314
- Full text: [PDF](../papers-all/2606.14674.pdf)
- Five-idea scores: PREQ-Harness=4; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce AGENT S PEC, a mod- not from the model alone, but from composing ular specification framework that represents these components into a coherent decision-making embodied agents as typed compositions of system. reusable policy components with standardized Yet despite their growing capabili...
- [introduction, p. 1] First, 2023), and self-correction (Madaan et al., 2023; we introduce AGENT S PEC, a typed modular spec- Shinn et al., 2023; Kumar et al., 2024), as well as ification for embodied LLM agents that separates memory mechanisms such as flat buffers (Zhong perception, memory, reasoning, reflection, action...
- [method, p. 8] For example, ReAct+DynamicCheatsheet, We introduced AGENT S PEC, a modular specifica- ReAct+MemoryBank, and ReAct+OpenClaw ob- tion framework for studying LLM-based embodied tain 5.02, 4.03, and 4.79, all below or comparable agents as typed compositions of perception, mem- to the simpler GRPO Base a...

**Result evidence**

- [results, p. 7] As matches long-horizon plan-then-act reasoning (see shown in Figure 6a, configurations with similar Appendix E.1.2 and Appendix E.1.3). token budgets can achieve very different profits, Multi-granularity memory is the safest default. while some higher-cost methods offer limited addi- MemoryBank per...
- [experiments, p. 5] As dimensions of embodied intelligence: Delivery- shown in Figure 3(a), stronger backbones gener- Bench (Mao et al., 2025) for long-horizon plan- ally achieve higher best-case DeliveryBench perfor- ning under resource constraints, ALFRED (Shrid- mance, with Qwen-27B and closed-source models 5 Figure...
- [experiments, p. 5] Full results are provided in Appendix E. outperforming smaller Qwen variants.

**Limitations / caution evidence**

- [related_work, p. 3] AGENT S PEC exposes the agent design space as a LLM-Based Agent Systems.
- [related_work, p. 3] Modern LLM agents controlled platform for analyzing component-level are often built as multi-step pipelines that inte- and interaction-level effects across tasks and back- grate reasoning, memory, tool use, reflection, and bones. action execution (Park et al., 2023; Hong et al., 2023; Chen et al., 2...
- [related_work, p. 3] Cognitive-inspired frameworks such as CoALA (Sumers et al., 2023) formalize agents as We instantiate AGENT S PEC as a Gym-compatible compositions of functional modules, while systems agent wrapper organized around a modular such as Voyager and AgentGym show that agents Perception–Memory–Reasoning–Re...

### [2606.01770: Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams](../reading-cards/2606.01770.md)

- Class: `core-selected`; pages: 23; local text characters: 83026
- Full text: [PDF](../papers-all/2606.01770.pdf)
- Five-idea scores: PREQ-Harness=4; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Adaptive Auto- bounded evolution grows from 12 to 34 skills, while the Harness, a framework and system for such prompt grows from 2 KB to 68 KB.

**Result evidence**

- [results, p. 2] All stopping budgets eventually harness evaluation is insufficient once tasks ar- peak and decline; later in the stream, shorter runs rive as unbounded, heterogeneous, and non- outperform longer ones.
- [conclusion, p. 9] Gepa: Reflec- by pairing sustained harness construction with tive prompt evolution can outperform reinforcement solve-time task adaptation.
- [abstract, p. 1] Domain-specific 100 40 Prompt size Auto-harness systems such as A-Evolve, Skills KB 50 20 arXiv:2606.01770v2 [cs.LG] 3 Jun 2026 GEPA, and Meta-Harness improve LLM agents by optimizing prompts, skills, tools, memo- 0 0 ries, and supporting infrastructure from exe- 0 news_from_future. md0 Skill Cycle...

**Limitations / caution evidence**

- [limitations, p. 9] We evaluate on three open- and efficient research flow. https://github.com/ ended task streams: prediction markets, cyberse- bytedance/deer-flow.
- [limitations, p. 9] Open-source software, ac- curity challenges, and event forecasting.
- [limitations, p. 9] These cessed 2026-05-28. domains cover unbounded streams, task hetero- Pu Cheng, Juncheng Liu, and Yunshen Long.

### [2606.06284: ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents](../reading-cards/2606.06284.md)

- Class: `core-selected`; pages: 18; local text characters: 43795
- Full text: [PDF](../papers-all/2606.06284.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We propose Causal Minimal Tool Filtering (CMTF), a training-free method that selects tools by causal sufficiency.
- [introduction, p. 1] We propose Causal Minimal Tool Filtering (CMTF), a training-free method for selecting tools based on causal sufficiency rather than semantic relevance alone.
- [introduction, p. 1] We introduce Causal Minimal Tool Filtering (CMTF), a lightweight, training-free filtering

**Result evidence**

- [experiments, p. 10] We evaluate whether causal tool filtering improves multi-step tool-use behavior under controlled conditions.
- [conclusion, p. 16] Across 2448 task-method-model runs with 102 tasks, 100 tools, four LLM backends, and six filtering strategies, CMTF achieved near-perfect aggregate success while substantially reducing wrong-tool calls, premature actions, visible tool exposure, and token usage.
- [conclusion, p. 16] The results show that simply reducing the number of tools is not sufficient: semantic top-k and state-aware filtering expose smaller tool sets but still underperform causal filtering.

**Limitations / caution evidence**

- [limitations, p. 16] This allows controlled evaluation of tool filtering under specified task states, goal states, and gold tool-chain annotations, but it may not capture the full ambiguity and variability of real tool ecosystems.
- [limitations, p. 16] In particular, the mocked tool outputs simplify environmental uncertainty such as tool failures, latency, partial results, authentication issues, and ambiguous observations.
- [limitations, p. 16] The experiments therefore primarily evaluate tool-selection behavior in simulated task environments rather than end-to-end robustness in deployed systems.

### [2405.15793: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](../reading-cards/2405.15793.md)

- Class: `core-selected`; pages: 118; local text characters: 292631
- Full text: [PDF](../papers-all/2405.15793.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=1

**Contribution evidence**

- [abstract, p. 1] As a result of this exploration, we introduce SWE-agent: a system that facilitates LM agents to autonomously use computers to solve software engineering tasks.
- [introduction, p. 1] From this effort, we introduce SWE-agent, an agent composed of an LM and ACI, that can interact with a computer to solve challenging real-world software engineering problems, such as those proposed in SWE-bench [20].
- [introduction, p. 1] First, we introduce the concept of the agent-computer interface (ACI) and demonstrate how careful ACI design can substantially improve LM agent performance without modifying the underlying LM’s weights.

**Result evidence**

- [abstract, p. 1] SWE-agent’s custom agent-computer interface (ACI) significantly enhances an agent’s ability to create and edit code files, navigate entire repositories, and execute tests and other programs.
- [abstract, p. 1] We evaluate SWE-agent on SWE-bench and HumanEvalFix, achieving state-of-the-art performance on both with a pass@1 rate of 12.5% and 87.7%, respectively, far exceeding the previous state-of-the-art achieved with non-interactive LMs.

**Limitations / caution evidence**

- [related_work, p. 9] 6.1 Software Engineering Benchmarks Code generation benchmarks, which evaluate models on the task of synthesizing code from natural language descriptions, have served as a long-standing bellwether for measuring LM performance [5, 1, 15, 30].
- [related_work, p. 9] Subsequent works have built upon the code generation task formulation to contribute new benchmarks that translate problems to different (programming) languages [3, 49], incorporate third-party libraries [25, 29], introduce derivative code completion tasks [18, 32], increase test coverage [26], chang...
- [related_work, p. 9] Code generation problems are largely self-contained, with short problem descriptions (∼100 lines) and corresponding solutions that are similarly brief, requiring nothing more complex than basic language primitives.

### [2604.03515: Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures](../reading-cards/2604.03515.md)

- Class: `core-selected`; pages: 42; local text characters: 142391
- Full text: [PDF](../papers-all/2604.03515.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=2; MRT-Harness=1; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [experiments, p. 25] Among these, what “memory” means varies significantly: Cline and Gemini CLI persist learned rules, Codex CLI extracts and consolidates memories from past sessions, while OpenCode persists full session state in SQLite, including all messages, tool outputs, token usage, and costs enabling interrupted...

**Limitations / caution evidence**

- [limitations, p. 33] This section organizes threats to validity following the standard framework for empirical software engineering studies [Runeson and Höst, 2009]: construct validity (whether the study measures what it claims to measure), internal validity (whether the findings follow from the data), external validit...
- [limitations, p. 33] 6.1 Construct Validity The primary construct validity threat is single-author bias.
- [limitations, p. 33] All 13 agent analyses were conducted by a single author (with LLM-assisted code navigation, as described in Section 3.4), meaning that dimension classifications, evidence selection, and cross-agent comparisons reflect one person’s interpretation of the source code.

### [2607.06906: The Harness Effect: How Orchestration Design Sets the Token Economics of Enterprise Agentic AI](../reading-cards/2607.06906.md)

- Class: `core-selected`; pages: 21; local text characters: 61149
- Full text: [PDF](../papers-all/2607.06906.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=1; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 12] Replacing the baseline with the harness removes 38% of token intensity and 41% of cost while quality stays at parity—which is the operational definition of escaping token maxing: the token exchange rate improved instead of degrading.
- [results, p. 12] The mass sits on or above the diagonal: 30 cells improve, 11 are flat, 7 regress.
- [results, p. 12] The frontier models and Palmyra improve most in exactly those categories (MCP: Sonnet +0.10, Palmyra +0.10; GDR: Sonnet +0.10, Palmyra +0.12).

**Limitations / caution evidence**

- [limitations, p. 17] Sample size. n = 22 prompts is sufficient for the uniform, large efficiency deltas and insufficient for quality inference; all quality claims are directional, and the headline is parity, not improvement.
- [limitations, p. 17] The baseline was run once and frozen (2026-06-07); run-to-run variance on the baseline is unmeasured.
- [limitations, p. 17] Task-completion is LLM-judged; judge bias is partially mitigated by locked criteria and secondary judges but not eliminated [20].

### [2605.26112: From Model Scaling to System Scaling: Scaling the Harness in Agentic AI](../reading-cards/2605.26112.md)

- Class: `core-selected`; pages: 14; local text characters: 53141
- Full text: [PDF](../papers-all/2605.26112.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Alongside the framework, we develop and release CheetahClaws2 , a Python-native reference harness, and use it together with Claude Code and OpenClaw as concrete points of comparison that make harness-level design choices explicit.
- [introduction, p. 3] We develop a systems-centered framing of agentic AI in which progress depends on scaling the harness, not only scaling the model.
- [introduction, p. 3] We propose a framework that separates base-model reasoning from system factors including memory, context construction, skill routing, orchestration, and verification-and-governance.

**Result evidence**

- [experiments, p. 11] We agree that model scaling will continue to improve agent behavior.
- [experiments, p. 11] End-to-end training may improve coordination across components, but deployed agents still require modular boundaries.

**Limitations / caution evidence**

- [conclusion, p. 11] Agentic AI is moving from isolated model inference to persistent system execution.
- [conclusion, p. 11] As models are embedded into tools, memory stores, repositories, browsers, subagents, and external services, their behavior is increasingly shaped by the architecture around them.
- [conclusion, p. 11] This paper has shown that future progress therefore requires system scaling: improving how agents construct context, maintain trustworthy memory, route skills, verify actions, govern tools, communicate across roles, and evolve over time.

### [2607.08716: Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents](../reading-cards/2607.08716.md)

- Class: `core-selected`; pages: 12; local text characters: 46661
- Full text: [PDF](../papers-all/2607.08716.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] We introduce a memory agent that runs alongside an unmodified action agent.
- [introduction, p. 1] Our contributions are: (i) we identify behavioral state decay as a central failure mode of long-horizon language agents; (ii) we propose a two-phase memory intervention architecture that decouples memory maintenance from action selection.
- [conclusion, p. 10] We propose agent memory as proactive intervention policy rather than passive storage and retrieval.

**Result evidence**

- [results, p. 7] Memory intervention improves pass@1 on both benchmarks and across both action-agent strengths: on Terminal-Bench 2.0, Sonnet 4.5 gains +8.3 pp (37.6% → 45.9%) and Opus 4.6 gains +2.4 pp (43.5% → 45.9%); on τ 2 -Bench, Sonnet 4.5 gains +6.8 pp on the task-weighted average (55.0% → 61.8%) and Opus 4.6...
- [results, p. 7] Table 2 shows that memory-style variants generally improve over the Sonnet baseline, but the improvements are not equally robust across domains.
- [results, p. 7] The full two-phase memory agent achieves the highest macro-average, improving all three domains and giving the largest gain on airline.

**Limitations / caution evidence**

- [conclusion, p. 10] We propose agent memory as proactive intervention policy rather than passive storage and retrieval.
- [conclusion, p. 10] Long- horizon agents suffer from behavioral state decay, where execution state that should guide future actions stops influencing behavior.
- [conclusion, p. 10] Our memory agent maintains this state and injects grounded context when they are likely to affect the next decision.

### [2607.05458: Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning](../reading-cards/2607.05458.md)

- Class: `core-selected`; pages: 17; local text characters: 63473
- Full text: [PDF](../papers-all/2607.05458.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=1; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] Learning is thus confined • We formulate the external control layer of a to the external control policy: the controller learns frozen LLM agent as a Harness MDP, in which how the agent should move through observation, a learned policy selects the next structural opera- evidence collection, verificat...

**Result evidence**

- [results, p. 7] Finding Result Evaluation Consistent improvement in veri- CheckBeforeSubmit increases from near-zero base rates to 5.6– Six controlled domains fication 17.8% across the controlled domains, 17.2% on τ -bench retail, and two benchmark and 16.7% on AgentBench DB-Bench. adapters Largest adapter-level ga...
- [results, p. 7] Positive transfer to AgentBench Final quality improves by 13.2% under the adapted deliberative- 16 training / 20 held-out DB-Bench reasoning rubric.
- [results, p. 7] Strongest controlled-domain Final quality improves by 10.0% under the calibrated structural 80 training / 20 held-out gain in coding verifier.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2210.03629: ReAct: Synergizing Reasoning and Acting in Language Models](../reading-cards/2210.03629.md)

- Class: `core-selected`; pages: 33; local text characters: 112020
- Full text: [PDF](../papers-all/2210.03629.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 8] ALFWorld, the best ReAct trial achieves an average success rate of 71%, significantly outperforming the best Act (45%) and BUTLER (37%) trials.
- [results, p. 8] With additional sparse reasoning, ReAct achieves significantly better performance, with an absolute 10% improvement over the previous best success rate.
- [results, p. 8] By checking examples, we find that ReAct is more likely to identify instruction-relevant products and options by reasoning to bridge the gap between noisy observations and actions (e.g. “For ‘space-saving ottoman bench for living room’, the item has options ‘39x18x18inch’ and ‘blue’ and seems good t...

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2605.13357: AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents](../reading-cards/2605.13357.md)

- Class: `core-selected`; pages: 16; local text characters: 50889
- Full text: [PDF](../papers-all/2605.13357.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We propose a different locus: software-engineering capability emerges from a model–harness–environment system, in which a runtime substrate—the harness— mediates how a foundation-model agent observes a project, acts on it, receives feedback, and establishes that a change is complete.
- [abstract, p. 1] We operationalize the harness through a four-level ladder (H0–H3) that progressively exposes runtime support to the agent, and we propose a trace-based
- [introduction, p. 1] Each of these is recognizable to practitioners and each maps, in our framework, to a missing harness responsibility.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2605.30785: Learning Agent-Compatible Context Management for Long-Horizon Tasks](../reading-cards/2605.30785.md)

- Class: `core-selected`; pages: 24; local text characters: 88566
- Full text: [PDF](../papers-all/2605.30785.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 5] Overall, AdaCoM consistently agement that led the agent to issue the successful improves over the vanilla ReAct baseline across all search.
- [results, p. 5] Second, SumCoM improves some agents but degrades GLM, suggest- Table 2: Mean@3 on MCP-Bench-Wiki across two agents.
- [experiments, p. 2] Instead of text management from the agent and learns building persistent memory across conversations, agent-compatible strategies without retraining we study working-memory management within a the agent itself. single long-horizon task, aiming to keep agents’ • AdaCoM substantially improves diverse...

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.24937: The Hitchhiker's Guide to Agentic AI: From Foundations to Systems](../reading-cards/2606.24937.md)

- Class: `direct-competitor`; pages: 603; local text characters: 1439412
- Full text: [PDF](../papers-all/2606.24937.pdf)
- Five-idea scores: PREQ-Harness=4; Harness Transport=2; MRT-Harness=0; Harness-C=26; ActiveHarness=3

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 230] The agent never forgets (library is persistent) and improves monotonically (only verified skills are added).
- [results, p. 230] A reward signal based on task success, user feedback, and efficiency metrics drives policy improvement.
- [abstract, p. 89] This exploits the model’s extensive pre-training on structured data (APIs, configs, code) to improve instruction adherence, reduce ambiguity, and enable deterministic parsing of multi-field requests.

**Limitations / caution evidence**

- [limitations, p. 103] A model can be confidently wrong (low entropy, consistent responses—but factually false).
- [limitations, p. 103] For reliable detection, combine with retrieval- based verification (RAG) or external fact-checking tools.
- [limitations, p. 103] 1.17 LLM Safety and Responsible AI Safety is not an afterthought—it is an integral part of the LLM training pipeline.

### [2607.01084: Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use](../reading-cards/2607.01084.md)

- Class: `direct-competitor`; pages: 51; local text characters: 136842
- Full text: [PDF](../papers-all/2607.01084.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=25; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] Conversely, RL frameworks such as grounding, remain vulnerable to boundary blindness due to ToolRL (Qian et al., 2025), DeepEyes (Zheng et al., 2026), a teleological bias in their reward structures. and others (Feng et al., 2025; Yu et al., 2025; Qian et al., Building on these insights, we introduce...

**Result evidence**

- [abstract, p. 1] Optimized via Supervised Fine-Tuning (SFT) demonstrate proficiency in static benchmarks, and Reinforcement Learning (RL), recent open-source mod- arXiv:2607.01084v1 [cs.AI] 1 Jul 2026 their deployment in real-world scenarios is hin- els (Hsieh et al., 2023; Qu et al., 2024; Qwen et al., 2024; dered...

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.24151: Metis: Bridging Text and Code Memory for Self-Evolving Agents](../reading-cards/2606.24151.md)

- Class: `direct-competitor`; pages: 18; local text characters: 69267
- Full text: [PDF](../papers-all/2606.24151.pdf)
- Five-idea scores: PREQ-Harness=25; Harness Transport=0; MRT-Harness=0; Harness-C=7; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2605.18747: Code as Agent Harness](../reading-cards/2605.18747.md)

- Class: `direct-competitor`; pages: 102; local text characters: 362267
- Full text: [PDF](../papers-all/2605.18747.pdf)
- Five-idea scores: PREQ-Harness=15; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=3

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 4] We further outline open challenges for harness engineering, including evaluation beyond final task success, verification under incomplete feedback, regression-free harness improvement, consistent shared state across multiple agents, human oversight, and extensions to multimodal environments.
- [experiments, p. 2] This layer shows how code becomes a shared harness for orchestrated autonomy: repositories, tests, traces, and structured artifacts provide the common workspace through which agents coordinate, inspect, and improve each other’s behavior.

**Limitations / caution evidence**

- [limitations, p. 43] Other QualityFlow [253]’s revert mechanism represents a synchronization pattern: the initial code artifact is never overwritten, enabling the system to roll back to a prior shared harness state if the debugging trajectory degrades quality.
- [limitations, p. 43] This is the only work among the surveyed system that explicitly manages state history rather than always moving forward.
- [limitations, p. 43] Position: The Shared Code-Centric Harness Substrate We propose a new position for the next generation of multi-agent intelligence: the shared code-centric harness substrate.

### [2605.17075: A Red Teaming Framework for Evaluating Robustness of AI-enabled Security Orchestration, Automation, and Response Systems](../reading-cards/2605.17075.md)

- Class: `direct-competitor`; pages: 15; local text characters: 51309
- Full text: [PDF](../papers-all/2605.17075.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=14; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce an autonomous red teaming framework that integrates large language models (LLMs) with reinforcement learning (RL) to generate adaptive, multi-stage attack cam- paigns against autonomous defenders in enterprise networks.
- [introduction, p. 1] To address these limitations, we propose a hierarchical hybrid framework that integrates LLM-based strategic planning with RL-based tactical execution for red team operations.
- [introduction, p. 1] We evaluate our approach in the Cyber Operations Research Gym (CybORG) CAGE Challenge 4 environ- ment [11, 12], a U.S. government–sponsored evaluation setting.

**Result evidence**

- [experiments, p. 7] Our experi- ments are designed to answer a key research question: whether integrating LLM-based strategic planning with RL-based execution improves autonomous red team performance against AI-enabled SOAR systems.
- [experiments, p. 7] Actions 0–3 perform reconnaissance at different stealth levels, actions 4–5 advance the kill chain through exploitation and privilege escalation, actions 6–7 achieve operational impact, and actions 8–9 provide evasive maneuvers.
- [experiments, p. 7] In particular, training terminates when the evaluation reward improves by less than 0.5% across five consecutive

**Limitations / caution evidence**

- [limitations, p. 2] 2 LITERATURE REVIEW This section reviews three bodies of literature that converge on the problem of autonomous red teaming: simulation environments for cyber operations, LLM applications in cybersecurity, and reinforcement learning for cyber agents.
- [limitations, p. 2] We then identify the gap that motivates our hierarchical architecture.
- [limitations, p. 2] 2.1 Cyber Operations Simulation Environments Evaluating autonomous cyber agents requires simulation environments that model realistic network topolo- gies, multi-agent interaction, and partial observability.

### [2605.14084: CRANE: Constrained Reasoning Injection for Code Agents via Nullspace Editing](../reading-cards/2605.14084.md)

- Class: `direct-competitor`; pages: 39; local text characters: 135416
- Full text: [PDF](../papers-all/2605.14084.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=13; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present CRANE (Con- strained Reasoning Injection for Code Agents via Nullspace Editing), a training-free parameter-editing method that treats the Thinking–Instruct delta as a directional pool of candidate reasoning edits for the Instruct backbone.

**Result evidence**

- [results, p. 9] metrics, while individual removals can improve isolated secondary metrics or cost.
- [experiments, p. 6] This section organizes the experiments around three research questions: RQ1: Does CRANE improve code-agent task success over the Instruct endpoint and standard merge baselines across IDE, 6 Table 1: Roo-Eval pass rates and token usage aggregated across five languages.
- [abstract, p. 1] By merging paired Instruct and Thinking checkpoints, CRANE de- livers strong gains over either individual model while preserving Instruct-level efficiency: on Roo-Eval it achieves pass@1 of 66.2% (+19.5%) for Qwen3-30B- A3B and 81.5% (+8.7%) for Qwen3-Next-80B-A3B; on SWE-bench-Verified it resolves...

**Limitations / caution evidence**

- [limitations, p. 9] First, CRANE assumes complementary paired endpoints: the Thinking checkpoint must provide useful reasoning behavior, and the Instruct checkpoint must define a useful deployment interface.
- [limitations, p. 9] If future Thinking models are already strong in task success, token efficiency, and tool discipline, a 9 simpler endpoint choice or global merge may be competitive.
- [limitations, p. 9] Second, the calibration sets must also cover the deployed tool surface; substantial drift in tools, formatting, or stopping behavior would require re-calibration.

### [2606.14502: From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI](../reading-cards/2606.14502.md)

- Class: `direct-competitor`; pages: 150; local text characters: 505924
- Full text: [PDF](../papers-all/2606.14502.pdf)
- Five-idea scores: PREQ-Harness=12; Harness Transport=0; MRT-Harness=0; Harness-C=8; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 18] ReAct [7] established the canonical form of this loop by interleaving Thought, Action, and Observation in an alternating chain, demonstrating that synergizing reasoning and acting outperforms either in isolation on knowledge-intensive and decision-making tasks.
- [results, p. 18] A dedicated line of GUI agents has pushed this further: CogAgent [652] employs a dual-resolution visual encoder for fine-grained UI recognition, ShowUI [653] unifies vision, language, and action in a single model that directly outputs UI operations from screenshots, and UI-TARS [654] achieves contex...
- [results, p. 18] The foundational insight came from Chain-of-Thought (CoT) prompting [15], which demonstrated that generating intermediate reasoning steps dramatically improves multi-step performance and now forms the backbone of most agent planning systems.

**Limitations / caution evidence**

- [limitations, p. 34] understanding next-generation agentic systems, it should not be interpreted as a complete solution to reliable autonomy.
- [limitations, p. 34] The paradigm improves continuity and reuse, but it also introduces new failure modes at the level of skills, workspaces, and their interaction.
- [limitations, p. 34] A balanced view is therefore necessary: persistent environments and reusable procedures can raise the ceiling of agentic work, but they also increase the need for lifecycle management, security review, and operational discipline [861, 862].

### [2606.27243: NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems](../reading-cards/2606.27243.md)

- Class: `direct-competitor`; pages: 12; local text characters: 65248
- Full text: [PDF](../papers-all/2606.27243.pdf)
- Five-idea scores: PREQ-Harness=10; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Across this trajectory, a clear pattern emerges: as simple causing silent failures that degrade performance. feature-engineering improvements become harder to obtain, fur- We present NOVA, a level-aware agent harness for verification- ther gains in recommendation quality and business impact increas-...
- [conclusion, p. 9] Dense and Sequence in Industrial Recommenders. arXiv preprint arXiv:2602.14110 We present NOVA, a level-aware agent harness for industrial recom- (2026). [8] Yunwen Huang, Shiyong Hong, Xijun Xiao, Jinqiu Jin, Xuanyuan Luo, Zhe Wang, mender architecture evolution.

**Result evidence**

- [results, p. 4] At each round, NOVA deliver the expected offline or online improvement.
- [abstract, p. 1] Across this trajectory, a clear pattern emerges: as simple causing silent failures that degrade performance. feature-engineering improvements become harder to obtain, fur- We present NOVA, a level-aware agent harness for verification- ther gains in recommendation quality and business impact increas-...
- [abstract, p. 1] Deployed in ibility, and validation through extensive offline-to-online pipelines. an industrial advertising system, NOVA achieves the highest effec- The process remains difficult to scale due to its operational com- tive pass rate on L2 ScaleUp and L3 Literature-to-Production tasks plexity in indus...

**Limitations / caution evidence**

- [conclusion, p. 9] Dense and Sequence in Industrial Recommenders. arXiv preprint arXiv:2602.14110 We present NOVA, a level-aware agent harness for industrial recom- (2026). [8] Yunwen Huang, Shiyong Hong, Xijun Xiao, Jinqiu Jin, Xuanyuan Luo, Zhe Wang, mender architecture evolution.
- [conclusion, p. 9] NOVA formulates architecture evo- Zheng Chai, Shikang Wu, Yuchao Zheng, and Jingjian Lin.
- [conclusion, p. 9] HyFormer: lution as a verification-aware architecture-gradient search, embeds Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR 9 Prediction. arXiv preprint arXiv:2601.12681 (2026).

### [2607.05377: Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation](../reading-cards/2607.05377.md)

- Class: `direct-competitor`; pages: 33; local text characters: 95808
- Full text: [PDF](../papers-all/2607.05377.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Cortex, a bidi- rectionally aligned embodied agent framework with a customized planning inter- face that conveys executable and tractable subtask plans from high-level VLM to low-level VLA.
- [introduction, p. 2] To tackle these challenges, we propose Cortex, a cognitive orchestrator aligned with execution.

**Result evidence**

- [experiments, p. 6] Crucially, Cortex achieves a state-of-the-art zero-shot success rate of 95.5%.
- [experiments, p. 6] 20 We attribute this robustness directly to our pro- posed ambiguity resolution mechanisms: (1) 0 ACT RDT-1BpenVLA-T DP3 0 X-VLA 0.5 ortex C urs) O OF (O Resolving Semantic Ambiguity: Fine-grained attribute and spatial grounding yield significant Figure 6: Success rates on RoboTwin benchmark.
- [experiments, p. 6] 7 improvements in visually dense tasks like place object basket (80% → 85%), conditioning the un- derlying VLA to predict kinematically favorable chunking.

**Limitations / caution evidence**

- [limitations, p. 9] Despite its robust planning capabilities, Cortex faces two primary limitations.
- [limitations, p. 9] Memory Repre- sentation: Text-based memory discards spatial coordinates and visual nuances, disrupting object- instance correspondence during large-scale mobile manipulation.
- [limitations, p. 9] Future work will integrate visual memory retrieval and pixel-level grounding to extend Cortex into a unified dual-mode framework.

### [2606.18356: SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents](../reading-cards/2606.18356.md)

- Class: `direct-competitor`; pages: 32; local text characters: 105508
- Full text: [PDF](../papers-all/2606.18356.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=3; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce SafeClawBench, a staged benchmark for tool-using agent security with 600 controlled adversarial tasks across six attack families: direct and indirect prompt injection, tool-return injection, memory poisoning, memory extraction, and ambiguity-driven unsafe inference.

**Result evidence**

- [abstract, p. 1] Evaluating five agent endpoints under four prompt-level policies, we find that these endpoints capture different failure modes.

**Limitations / caution evidence**

- [limitations, p. 10] SafeClawBench is a controlled stress-test suite, so its numbers are best read as comparative endpoint measurements rather than as operational incident rates.
- [limitations, p. 10] The prompt-policy matrix isolates one important control layer; deployment systems should combine these policies with runtime tool permissions and monitoring.
- [limitations, p. 10] Some harms that require long-horizon context or external services are outside the current sandbox.

### [2605.24117: SkillEvolBench: Benchmarking the Evolution from Episodic Experience to Procedural Skills](../reading-cards/2605.24117.md)

- Class: `direct-competitor`; pages: 42; local text characters: 154426
- Full text: [PDF](../papers-all/2605.24117.pdf)
- Five-idea scores: PREQ-Harness=4; Harness Transport=3; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce SkillEvolBench, a diagnostic benchmark for evaluating this step from experience reuse to skill formation.
- [introduction, p. 1] To study this question, we introduce SkillEvolBench, a diagnostic benchmark for the missing step between episodic experience and procedural reuse.
- [conclusion, p. 14] We introduced SkillEvolBench, a diagnostic benchmark for evaluating whether agents can transform episodic task experience into reusable procedural skills.

**Result evidence**

- [experiments, p. 7] We interpret episodic experience as having become a reusable skill only when the resulting library improves not just the original acquisition or replay tasks, but also frozen deployment tasks that require invocation, robustness, and composition.
- [experiments, p. 7] Positive values indicate that distilled skills outperform direct episodic reuse, while negative values indicate that raw trajectories preserve useful task evidence lost during skill abstraction.
- [experiments, p. 7] Red indicates improvement, blue indicates degradation, and gray indicates unavailable comparisons. based conditions can improve LSR or RSR, and some model-condition pairs achieve strong gains on specific deployment axes.

**Limitations / caution evidence**

- [conclusion, p. 14] We introduced SkillEvolBench, a diagnostic benchmark for evaluating whether agents can transform episodic task experience into reusable procedural skills.
- [conclusion, p. 14] Rather than only measuring whether skills help at inference time, SkillEvolBench targets the missing step from experience reuse to skill formation.
- [conclusion, p. 14] Its role-conditioned task families, verifier-backed feedback, frozen deployment phase, replay setting, and Raw-Trajectory control help separate local task recovery from transferable procedural reuse.

### [2605.02801: Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces](../reading-cards/2605.02801.md)

- Class: `direct-competitor`; pages: 71; local text characters: 211132
- Full text: [PDF](../papers-all/2605.02801.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=5; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 38] We found no general fix in our retained pool; debate- style topologies [51] partly sidestep by making diversity part of the reward. • Over-communication (O3).
- [results, p. 38] This can be methodologically hazardous: single-agent benchmarks measure task success, which any system with enough compute can improve; they do not measure whether the improvement came from genuine multi-agent coordination.
- [results, p. 38] Latent- MAS [84] is the clearest evidence that much of (E3) can be achieved without natural- language messaging at all. • (E4) Protocol overhead.

**Limitations / caution evidence**

- [limitations, p. 48] This survey is intended as a curated taxonomy and position paper, not as an exhaustive sys- tematic review.
- [limitations, p. 48] Four limitations are therefore important for interpreting its claims.
- [limitations, p. 48] The retained pool contains 84 entries se- lected for their relevance to reward design, credit assignment, orchestration learning, systems constraints, benchmarks, or safety in LLM-MAS RL.

## measurement-evaluation

### [2210.01948: Game-theoretic statistics and safe anytime-valid inference](../reading-cards/2210.01948.md)

- Class: `core-selected`; pages: 25; local text characters: 143495
- Full text: [PDF](../papers-all/2210.01948.pdf)
- Five-idea scores: PREQ-Harness=46; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=10

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2605.23950: Stop Comparing LLM Agents Without Disclosing the Harness](../reading-cards/2605.23950.md)

- Class: `core-selected`; pages: 17; local text characters: 64023
- Full text: [PDF](../papers-all/2605.23950.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=15; MRT-Harness=0; Harness-C=17; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Third, we propose a harness-aware evaluation framework with a disclosure standard and a variance decomposition protocol.

**Result evidence**

- [experiments, p. 8] The empirical trend goes the other way: the harness ecosystem has grown more complex as models have improved, with Claude Code redesigning rather than removing harness components, the managed- agents architecture explicitly decoupling model and harness, and AHE producing double-digit gains on top of...
- [experiments, p. 8] Harness improvements are real gains that deployments rightly capture.
- [abstract, p. 1] We formalize and defend the Binding Con- straint Thesis: in this regime, performance variance is governed more by harness configuration than by model choice, and current evaluation protocols therefore systematically misattribute harness-level gains to model improvements.

**Limitations / caution evidence**

- [conclusion, p. 9] For researchers, reviewers should ask “what harness was used?” as routinely as they ask about hyperparameters and decoding settings, since a model score without a harness specification is missing part of the experimental condition.
- [conclusion, p. 9] For benchmark designers, agent benchmarks should expose harness variation as a first-class evaluation dimension, either through locked-harness tracks for clean model comparison or factorial tracks that report variance decompositions, building on programs underway in HAL [14], the unified scaffold of...
- [conclusion, p. 9] For practitioners, model selection alone is an incomplete optimization loop: in our controlled grid, moving from H1 to H3 shifts pass@1 by 8.5 to 13.0 percentage points at a fixed model, while changing the model at a fixed harness shifts pass@1 by 2.5 to 5.0 points.

### [2605.02122: STABLEVAL: Disagreement-Aware and Stable Evaluation of AI Systems](../reading-cards/2605.02122.md)

- Class: `core-selected`; pages: 19; local text characters: 65847
- Full text: [PDF](../papers-all/2605.02122.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=16; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] We introduce STABLEVAL, a disagreement-aware ies evaluating whether large language models can capture

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [related_work, p. 2] STABLEVAL Framework growing interest in treating annotator disagreement as a We consider the problem of evaluating a set of AI agents meaningful signal rather than noise (Fleisig et al., 2023). using noisy and heterogeneous human judgments.
- [related_work, p. 2] In con- Surveys and frameworks highlight the need to model struc- trast to classical annotation aggregation settings, where tured disagreement, especially for subjective tasks like toxi- the primary goal is to recover a single consensus or de- city detection and stance annotation (Xu & Jurgens, 2026...
- [related_work, p. 2] Methods that explicitly model annotator-specific annotator variability and disagreement (Benito-Santos & behavior or group-level variation via demographic-aware Ghajari, 2025). experts and synthetic perspectives demonstrate improved representation of structured disagreement (Xu et al., 2025; Standar...

### [2603.23749: Efficient Benchmarking of AI Agents](../reading-cards/2603.23749.md)

- Class: `core-selected`; pages: 22; local text characters: 63742
- Full text: [PDF](../papers-all/2603.23749.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=15; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Exploiting this asymmetry, we propose a simple optimization-free protocol: evaluate new agents only on tasks with intermediate historical pass rates (30–70%).
- [introduction, p. 2] We propose the Mid-Range Difficulty Filter (MR), a deterministic, optimization-free task selection rule that retains tasks with pass rates between 30–70%.

**Result evidence**

- [results, p. 8] 4.1 The ρ–R2 Divergence: Rank Prediction is Preserved Even When Score Prediction Collapses By running 5 evaluation protocols across 8 agent benchmarks and 6 task selection strategies, we show empirically that leaderboard rankings are preserved better than absolute scores.
- [conclusion, p. 13] Our results show that reliable AI agent leaderboard rankings do not require full-benchmark evalu- ation.
- [abstract, p. 1] Across eight benchmarks, 33 agent scaffolds, and 70+ model configurations, we find that absolute score prediction degrades under this shift, while rank-order prediction remains stable.

**Limitations / caution evidence**

- [limitations, p. 13] MR requires a populated mid-range difficulty band to function properly.
- [limitations, p. 13] It fails on SciCode, where only about four tasks fall into this band, and more generally, benchmarks with highly skewed difficulty distributions are less compatible with this approach.
- [limitations, p. 13] There is a non-trivial cold-start cost: approximately five to ten agents must be evaluated on the full benchmark before reduction becomes reliable for incremental use, and closer to fifteen agents are needed for stability across all future comparisons.

### [2407.01502: AI Agents That Matter](../reading-cards/2407.01502.md)

- Class: `core-selected`; pages: 33; local text characters: 116619
- Full text: [PDF](../papers-all/2407.01502.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=13; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We design and implement one such optimization, showing its potential to greatly reduce cost while maintaining accuracy.
- [abstract, p. 1] We hope that the steps we introduce for addressing these shortcomings will spur the development of agents that are useful in the real world and not just accurate on benchmarks.
- [introduction, p. 1] We introduce three new simple baseline agents and empirically * Equal Contribution.

**Result evidence**

- [results, p. 4] Note the nonstandard axes; In Appendix A, we show our results with the full y-axis as well as error bars and provide additional details.
- [results, p. 4] Meanwhile, the escalation strategy strictly improves accuracy while costing less than half of LDB (GPT-3.5).
- [results, p. 4] Accuracy alone cannot identify progress because it can be improved by scientifically meaningless methods such as retrying.

**Limitations / caution evidence**

- [conclusion, p. 11] AI agent benchmarking is new and best practices haven’t yet been established, making it hard to distinguish genuine advances from hype.
- [conclusion, p. 11] Our thesis is that agents are sufficiently different from models that benchmarking practices need to be rethought.
- [conclusion, p. 11] We have taken the first steps toward a principled approach to agent benchmarking, resulting in recommendations including cost-controlled comparisons, separating model and downstream evaluation, preventing shortcuts using appropriate hold-outs, and greater standardization of evaluation practices.

### [2606.15508: ToolMenuBench: Benchmarking Tool-Menu Filtering Strategies for Reliable and Efficient LLM Agents](../reading-cards/2606.15508.md)

- Class: `core-selected`; pages: 13; local text characters: 50887
- Full text: [PDF](../papers-all/2606.15508.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=2; MRT-Harness=0; Harness-C=12; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce ToolMenuBench, a benchmark for evaluating tool-menu construction in multi-step LLM agents.
- [abstract, p. 1] This paper makes the following contributions: 1) We introduce ToolMenuBench, a benchmark for evaluating how visible tool-menu construction affects reliability, effi- ciency, and safety-relevant exposure in multi-step LLM agents.

**Result evidence**

- [abstract, p. 1] In a controlled evaluation across seven model backends, three tool-menu sizes, six filtering methods, and seven evaluation settings, CMTF improves task success from 32.1% under all-tools exposure to 85.7%, while reducing average token usage by roughly 98%.
- [abstract, p. 1] Causal minimal tool filtering achieves the strongest overall tradeoff, reducing visible tools, wrong-tool calls, premature actions, and risky-tool exposure relative to unfiltered exposure, lexical filtering, state-aware filtering, and broader causal-path baselines.
- [abstract, p. 1] 4) We report a controlled core evaluation across multiple model backends, tool-menu sizes, filtering methods, and distractor settings, showing that causally aligned menus can improve success while reducing tool exposure and token usage.

**Limitations / caution evidence**

- [related_work, p. 2] Tool-Augmented LLM Agents Tool use has become a central mechanism for extending large language models beyond text generation.
- [related_work, p. 2] ReAct introduced interleaved reasoning and acting, allowing models to combine language-based reasoning with external actions [1].
- [related_work, p. 2] Toolformer showed that language models can learn to call external APIs during inference [2].

### [2605.27922: Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows](../reading-cards/2605.27922.md)

- Class: `core-selected`; pages: 16; local text characters: 51253
- Full text: [PDF](../papers-all/2605.27922.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Harness-Bench, a diagnostic benchmark for evaluating configuration-level harness effects in realistic agent workflows.
- [introduction, p. 1] We introduce Harness-Bench, a diagnostic benchmark for studying configuration-level harness effects in realistic agent workflows.
- [introduction, p. 1] We introduce Harness-Bench, a suite of 106 sandboxed offline tasks for evaluating realistic end-to-end agent workflows with task manifests, fixtures, evaluators, and execution traces. (2) Evaluation protocol.

**Result evidence**

- [conclusion, p. 9] Across 5,194 trajectories, we observe substantial differences across model–harness configurations, supporting the need to report agent capability at the configuration level rather than by the base model alone.
- [conclusion, p. 9] We hope Harness-Bench helps diagnose and improve reliable, efficient, permission-aware, and auditable agent execution stacks.
- [abstract, p. 1] Across 5,194 execution trajectories, we observe substan- tial variation in completion, process quality, efficiency, and failure behavior across model–harness pairings.

**Limitations / caution evidence**

- [conclusion, p. 9] We presented Harness-Bench, a diagnostic benchmark for studying configuration-level harness effects in realistic executable agent workflows.
- [conclusion, p. 9] By fixing external task conditions while preserving each harness’s native execution behavior, Harness-Bench makes execution-layer variation observable under a shared protocol.
- [conclusion, p. 9] Across 5,194 trajectories, we observe substantial differences across model–harness configurations, supporting the need to report agent capability at the configuration level rather than by the base model alone.

### [2606.19613: StaminaBench: Stress-Testing Coding Agents over 100 Interaction Turns](../reading-cards/2606.19613.md)

- Class: `core-selected`; pages: 37; local text characters: 124510
- Full text: [PDF](../papers-all/2606.19613.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=7; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce StaminaBench, a benchmark that measures the stamina of coding agents: how many consecutive interaction turns (change requests) they can handle before failing.
- [introduction, p. 1] We introduce StaminaBench to stress-test these capabilities.

**Result evidence**

- [results, p. 6] Bold are the best per row, underlined are the results that are not significantly worse than the best under an uncorrected Wilcoxon signed-rank test (p > 0.05).
- [results, p. 6] All 26 model×harness cells improve significantly over R = 0 (Wilcoxon signed- rank, Holm-corrected, p < 0.05).
- [results, p. 6] Provider-built harnesses do not reliably help: QwenCode is significantly worse than OpenCode for both Qwen3-Coder-Next (p = 0.014) and Qwen3.5-122B (p = 0.011), while other provider pairings show good averages but no significant difference from OpenCode.

**Limitations / caution evidence**

- [related_work, p. 3] SWE-Bench [23] and its verified subset present real GitHub issues for agents to resolve and have become the de facto standard for evaluating coding agents.
- [related_work, p. 3] Most recent models achieve close to 80% on it, prompting follow-ups [11, 58, 59, 61].
- [related_work, p. 3] HumanEval [9] and MBPP [4] evaluate function-level code generation.

### [2311.12983: GAIA: a benchmark for General AI Assistants](../reading-cards/2311.12983.md)

- Class: `core-selected`; pages: 24; local text characters: 82921
- Full text: [PDF](../papers-all/2311.12983.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=7; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] We hope our methodology will help addressing the problem of open ended generation evaluation in NLP and beyond, and believe the successful resolution of GAIA would be an important milestone towards the next generation of AI systems.

**Result evidence**

- [results, p. 9] 0.075 The discrepancy between GPT4 results with- 0.050 out plugins and the others demonstrate that 0.025 augmenting LLMs via tool APIs or access to 0.000 the web improves answer accuracy, and un- readin g N/A owsin g Codin g ality etype Web br i-mod lock many new use cases, confirming the e fil Mult...
- [results, p. 9] Moving forward, the conjugation of multi-modal systems with GAIA might further improve advanced generative models evaluation e.g. image generators, via tasks requiring a complex sequence of image modifications and asking an unambiguous question on the resulting image in natural language.
- [results, p. 9] Full automation is a goal that deep learning has been striving to achieve, without complete success to date: in spite of state-of-art results in various domains, most neural networks based systems can unpredictably fail e.g in common situations, impeding the advent of technologies such as self-drivi...

**Limitations / caution evidence**

- [limitations, p. 3] than the one currently evaluated, and the quality of the evaluation is affected by the shortcomings of the evaluator LLM, which are not always obvious and can lead to subtly incorrect results.
- [limitations, p. 3] While there is ongoing effort to turn Large Language Models into general- purpose assistants (see our discussion in Appendix A), appropriate evaluation is lagging behind.
- [limitations, p. 3] Most evaluations rely on the use of closed systems, specific API calls, and a given “correct way” to attain the answer, or simply repurpose existing evaluation datasets.

### [2601.10343: OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding](../reading-cards/2601.10343.md)

- Class: `core-selected`; pages: 21; local text characters: 80935
- Full text: [PDF](../papers-all/2601.10343.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=5; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Median checklist items per instance 34 To fill this gap, we introduce O CTO B ENCH, which benchmarks scaffold-aware instruction Table 1: Overall statistics of O CTO B ENCH. following in repository-grounded agentic cod- ing.
- [introduction, p. 1] As a result, an agent rapidly in recent years, enabling increasingly ca- may appear correct while silently breaking higher- pable reasoning and tool use across a wide range priority constraints. of applications (MiniMax et al., 2025; Seed et al., To address this gap, we introduce O CTO B ENCH, 2025;...
- [conclusion, p. 8] This performance decay suggests that most mod- We introduce OctoBench to evaluate how models els experience context fatigue during protracted follow heterogeneous instructions in agentic coding workflows.

**Result evidence**

- [results, p. 6] Gemini-3-Pro (Sundar Pichai et al.)) and report the ensemble-averaged results to mitigate potential Finding 2: Model performance in instruction judge bias. following varies significantly depending on the instruction category.
- [results, p. 6] Higher Metrics are in %. ∆ is the absolute improvement. values indicate stronger adherence to that source.
- [experiments, p. 3] De- 3 O CTO B ENCH spite improved realism, most evaluations remain outcome-oriented, providing limited visibility into 3.1 Datasets whether solutions satisfy non-functional or process constraints (Singhal et al., 2024; Shen et al., 2025).

**Limitations / caution evidence**

- [conclusion, p. 8] This performance decay suggests that most mod- We introduce OctoBench to evaluate how models els experience context fatigue during protracted follow heterogeneous instructions in agentic coding workflows.
- [conclusion, p. 8] Our results show that agents often fail to significant outlier by maintaining high adherence maintain long-term instruction ability even when capabilities even as conversation length increases, they successfully complete a task.
- [conclusion, p. 8] We identify demonstrating a level of long-horizon robustness a major gap between passing individual checks that is absent in other evaluated models. and maintaining overall reliability, especially when models must resolve conflicting rules or follow 4.3.4 RQ5: Is the LLM-as-a-Judge Evaluation comple...

### [2604.11978: The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break](../reading-cards/2604.11978.md)

- Class: `core-selected`; pages: 47; local text characters: 133599
- Full text: [PDF](../papers-all/2604.11978.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To address this gap, we introduce HORIZON, an initial cross-domain diagnostic bench- mark for systematically constructing tasks and analyzing long-horizon failure behaviors in LLM-based agents.
- [conclusion, p. 9] In this work, we introduced HORIZON, an initial cross-domain diagnostic benchmark for systematically constructing long-horizon tasks and analyzing failure trajectories.

**Result evidence**

- [results, p. 2] These findings suggest that scaling base models alone is insufficient; robust long-horizon performance requires method-level improvements in planning, memory, and execution-time control.
- [experiments, p. 9] Although some long-horizon failures can be at- tributed to imperfect base-model capabilities, our results suggest that model scaling alone is unlikely to resolve the dominant failure mechanisms.
- [conclusion, p. 9] Across 3100+ trajectories and multiple model families, we showed that long-horizon breakdown is 9 Preprint.

**Limitations / caution evidence**

- [limitations, p. 44] OS (SR 40.3%) has by far the most diverse failure profile: Planning Error (36.7%), Instruction (25.9%), Environment (17.3%), and Memory Limitation (15.1%) all contribute substantially, consistent with the open-ended, long-horizon nature of shell tasks.
- [limitations, p. 44] History Error Accumulation appears exclusively in OS (0.1%).
- [limitations, p. 44] 100% 80% % of Failed Traces 60% 40% 20% 0% GPT-4o-mini Claude 3.5 Sonnet Failure Type: Environment Planning Error Memory Limitation False Assumption Instruction History Error Accumulation Catastrophic Forgetting Figure 7: Failure mode distribution for GPT-4o-mini vs.

### [2604.02460: Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](../reading-cards/2604.02460.md)

- Class: `core-selected`; pages: 27; local text characters: 75564
- Full text: [PDF](../papers-all/2604.02460.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=2; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present an information- theoretic argument, grounded in the Data Processing Inequality, suggesting that under a fixed reasoning-token budget and with perfect context utiliza- tion, single-agent systems are more information-efficient.
- [conclusion, p. 10] We presented a budget-controlled comparison of single-agent (SAS) and multi-agent (MAS) LLM systems, focusing on fixed thinking token budgets.

**Result evidence**

- [conclusion, p. 10] Our results across two datasets (FRAMES, MuSiQue), three model families (Qwen3, DeepSeek, Gemini), and five different MAS architectures (Sequential, Debate, Ensemble, Parallel-roles, Subtask- parallel) consistently show that SAS matches or outperforms MAS when computation is normalized, unless conte...
- [conclusion, p. 10] Overall, our results suggest that many reported MAS gains are better explained by compute and context effects than by inherent architectural superiority, and that future work should focus on the specific regimes where multi-agent structure provides real benefit.
- [abstract, p. 1] When computation is normalized, single-agent systems (SAS) can match or outperform MAS, yet the theoretical basis and evaluation method- ology behind this comparison remain unclear.

**Limitations / caution evidence**

- [conclusion, p. 10] We presented a budget-controlled comparison of single-agent (SAS) and multi-agent (MAS) LLM systems, focusing on fixed thinking token budgets.
- [conclusion, p. 10] Our results across two datasets (FRAMES, MuSiQue), three model families (Qwen3, DeepSeek, Gemini), and five different MAS architectures (Sequential, Debate, Ensemble, Parallel-roles, Subtask- parallel) consistently show that SAS matches or outperforms MAS when computation is normalized, unless conte...
- [conclusion, p. 10] Overall, our results suggest that many reported MAS gains are better explained by compute and context effects than by inherent architectural superiority, and that future work should focus on the specific regimes where multi-agent structure provides real benefit.

### [2406.12045: $τ$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](../reading-cards/2406.12045.md)

- Class: `core-selected`; pages: 50; local text characters: 131158
- Full text: [PDF](../papers-all/2406.12045.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We propose τ -bench, a benchmark emulating dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines.
- [introduction, p. 1] In this work, we introduce τ -bench (short for Tool-Agent-User Interaction Benchmark) to measure an agent’s ability to interact with (simulated) human users and programmatic APIs while following domain-specific policies in a consistent manner. τ -bench is built in a modular framework with (1) realis...

**Result evidence**

- [results, p. 6] Notably, SoTA open-weight models (llama-3-70b and mistral-8x22b) still have a significant gap to cover with respect to SoTA 6 Model retail airline avg 60 FC gpt-4o 61.2 35.2 48.2 40 ReAct gpt-4-turbo 57.7 32.4 45.1 Act gpt-4-32k 56.5 33.0 44.8 20 gpt-3.5-turbo 20.0 10.8 15.4 0 claude-3-opus 44.2 34....
- [abstract, p. 1] Our experiments show that even state-of-the-art function calling agents (like gpt-4o) succeed on < 50% of the tasks, and are quite inconsistent (pass^8 < 25% in retail).
- [abstract, p. 1] Our findings point to the need for methods that can improve the ability of agents to act consistently and follow rules reliably.

**Limitations / caution evidence**

- [related_work, p. 2] Most existing benchmarks for agents and task-oriented dialogue systems focus on evaluating either conversational or tool-use capabilities. τ -bench aims to unify both under realistic settings, while also testing how well agents can follow domain-specific policies in a consistent manner.
- [related_work, p. 2] Several benchmarks have been developed to evaluate agents powered by LMs [27, 29, 12, 14, 16] .
- [related_work, p. 2] Recent efforts have focused specifically on evaluating tool use capabilities of LMs, i.e., their ability to generate the right function calls from a set of functions in an API.

### [2310.06770: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../reading-cards/2310.06770.md)

- Class: `core-selected`; pages: 52; local text characters: 154416
- Full text: [PDF](../papers-all/2310.06770.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=1; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [method, p. 8] The total model input consists of 1,558 lines of context or 20,882 to- kens.
- [conclusion, p. 26] C.5 F2P, P2P R ATE A NALYSIS In the main paper results, we present the “% Resolved” statistic that indicates how many task in- stances were completely solved by the different models.
- [conclusion, p. 26] The difference relative to percentages in Table 5 and Table 18 is included as a subscript. established, we introduce five new terms.

**Result evidence**

- [results, p. 18] We find that release versions are a good proxy for capturing the dependency requirements across a subset of task instances, striking a manageable balance between installation success and manual effort.
- [results, p. 18] Based on the source code and documentation typically found in the repository’s README and CONTRIBUTING guides, we find out the Python version, necessary dependencies, and installation command.
- [results, p. 18] With moderate variation across repositories, we observe that this step generally removes half of the candidate task instances.

**Limitations / caution evidence**

- [limitations, p. 10] SWE-bench’s task instance collection procedure to expand its coverage to more programming lan- guages and domains.
- [limitations, p. 10] Second, our experiments aim to establish a baseline of the simplest and most straight-forward approaches for this task; we do not intend to constrain future methodologies to the same type of approach and encourage future work to investigate different methods (e.g., agent-based approaches, tool augme...
- [limitations, p. 10] Lastly, while this work evaluates models using execution-based code testing, relying solely on this

### [2307.13854: WebArena: A Realistic Web Environment for Building Autonomous Agents](../reading-cards/2307.13854.md)

- Class: `core-selected`; pages: 22; local text characters: 78666
- Full text: [PDF](../papers-all/2307.13854.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [experiments, p. 16] We aim to design tasks that have easy-to-imagine outcomes (e.g., a new product page is created) rather than those that are easily performed by an average user without significant domain knowledge.
- [experiments, p. 16] Issue stop action when you think you have achieved the objective.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.12882: HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness](../reading-cards/2606.12882.md)

- Class: `core-selected`; pages: 29; local text characters: 96330
- Full text: [PDF](../papers-all/2606.12882.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 2] Thus, we propose HarnessBridge, a learnable harness policy for long-horizon LLM agents.
- [introduction, p. 2] Our contributions are as fllows: • We introduce end-to-end harness generation for agent systems, replacing manually engineered interaction logic with a learnable harness policy. • We are the first to introduce unified instruction tuning for learning bidirectional mappings between agents and environm...

**Result evidence**

- [results, p. 7] Overall, HarnessBridge jointly improves success rate and reduces token consumption compared to other harness design.
- [results, p. 7] Nevertheless, HarnessBridge still achieves strong performance, suggesting that the learned harness policy generalizes beyond the training environment.
- [results, p. 7] On Terminal-Bench 2.0, HarnessBridge achieves the highest success rate under both generators, reaching 33.7% with Qwen3.5-35B-A3B and 20.7% with GLM-4.7-Flash.

**Limitations / caution evidence**

- [conclusion, p. 10] HarnessBridge recasts the agent–environment interface as a learnable harness policy.
- [conclusion, p. 10] Through jointly trained bidirectional projections, it compresses raw trajectories into decision-critical agent context and maps proposed actions into executable transitions or trajectory-grounded rejections.
- [conclusion, p. 10] HarnessBridge achieves competitive or stronger performance with lower token cost and shorter trajectories on the benchmarks, while transferring from small to larger commercial models.

### [2601.11868: Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](../reading-cards/2601.11868.md)

- Class: `core-selected`; pages: 84; local text characters: 208001
- Full text: [PDF](../papers-all/2601.11868.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [method, p. 2] Tasks are specified using the Harbor task format and are run using the Harbor harness, which supports popular agents, including Claude Code, Codex CLI, OpenHands, and Mini-SWE-Agent, as well as our own agent, Terminus 2, which we developed as a neutral testbed for comparing model performance (Sectio...

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [limitations, p. 26] <task.yaml> {task_yaml} </task.yaml> <run_info> {run_info} </run_info> <trials_data> {trials_data} </trials_data>

### [2404.07972: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](../reading-cards/2404.07972.md)

- Class: `core-selected`; pages: 51; local text characters: 157691
- Full text: [PDF](../papers-all/2404.07972.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To address this issue, we introduce OSW ORLD, the first-of-its-kind scalable, real computer environment for multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across various operating sys- tems such as Ubuntu, Windows, and macOS.
- [introduction, p. 1] To address this gap, we introduce OSW ORLD, the first-of-its-kind scalable, real computer environment designed for the development of multimodal agents capable of executing a wide range of real computer tasks beyond isolated interfaces and applications.

**Result evidence**

- [results, p. 11] Overall, these figures of performance are significantly lower than the human-level performance which is 72.36% overall for individuals not familiar with the software.
- [results, p. 11] These gaps indicate that current LLMs and VLMs may still have a significant gap from humans in performance, necessitating further research in this area.
- [results, p. 11] We observe performance based on software type grouping and find that agents based on LLMs show significant differences across different subsets.

**Limitations / caution evidence**

- [conclusion, p. 17] In conclusion, the introduction of OSW ORLD marks a significant step forward in the development of autonomous digital agents, addressing critical gaps in existing interactive learning environments.
- [conclusion, p. 17] By providing a rich, realistic setting that spans multiple operating systems, interfaces, and applications, OSW ORLD not only broadens the scope of tasks digital agents can perform but also enhances their potential for real-world application.
- [conclusion, p. 17] Despite the promise shown by advancements in vision-language models, evaluations within OSW ORLD reveal notable challenges in agents’ abilities, particularly in GUI understanding and operational knowledge, pointing to essential areas for future research and development.

### [2602.22953: General Agent Evaluation](../reading-cards/2602.22953.md)

- Class: `core-selected`; pages: 50; local text characters: 165122
- Full text: [PDF](../papers-all/2602.22953.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=1; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [conclusion, p. 33] Closed-source models swing 7–12pp; open-weight models swing 14–18pp with highly significant best-vs-worst tests.
- [abstract, p. 1] We find that (i) general agents adapt to every tested domain without per-domain customization; (ii) agent architecture choice swings

**Limitations / caution evidence**

- [limitations, p. 14] 14 A Framework and Adaptation Details This appendix details how we adapt existing benchmarks and agents to the Unified Protocol, and the Exgentic orchestrator design that runs the full factorial evaluation.
- [limitations, p. 14] Table 4 reproduces the comparison table from the main paper (Table 1) with the full caption defining each of the five axes used to position this work against prior evaluation systems.
- [limitations, p. 14] Table 4: Positioning of this work relative to prior agent-evaluation studies and frameworks along five × axes. ✓ indicates the property holds; ✗ indicates it does not; ✓ indicates it holds within a constrained class.

### [2410.07095: MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](../reading-cards/2410.07095.md)

- Class: `core-selected`; pages: 29; local text characters: 89673
- Full text: [PDF](../papers-all/2410.07095.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 15] We observe that no submission has a similarity score above 60%.
- [results, p. 15] In the following, we describe any further scaffold-specific modifications to address common pitfalls and improve the robustness of the agents.
- [results, p. 15] A.6.1 AIDE MODIFICATIONS • Implement exponential backoff on API call rate limits to handle high traffic scenarios. • Add strict: True to Function Calling outputs to ensure stricter enforcement of out- put format rules, preventing invalid feedback responses. • Add support for the Gemini and OpenRoute...

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2604.00594: Agent psychometrics: Task-level performance prediction in agentic coding benchmarks](../reading-cards/2604.00594.md)

- Class: `core-selected`; pages: 36; local text characters: 105747
- Full text: [PDF](../papers-all/2604.00594.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=1; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [method, p. 8] However, we remind the reader that both the LLM and the scaffold have individually been represented in the training data; our method cannot generalize to entirely new LLMs or scaffolds.
- [method, p. 8] The results show that for lower compute budgets under 30 tasks, our method outperforms randomly selecting a subset.
- [method, p. 8] As a result of this decomposition, we obtain a novel quantitative ranking of scaffold abilities, which we present in Table 14 in Appendix G.2.

**Result evidence**

- [experiments, p. 18] We use DeepSeek-R1-Distill-Qwen-32B (Guo et al., 2025) as the embedding backbone, as we found it to produce the best embeddings among the 17 backbones we tested (see Appendix D.1).
- [experiments, p. 18] We extract at most 7 features per API call, as we found that with more features, Claude Opus 4.6 tends to leave out features from its response.

**Limitations / caution evidence**

- [related_work, p. 3] 2.1.1 AGENTIC C ODING B ENCHMARKS In contrast to question-and-answer benchmarks, agentic benchmarks require the LLM to dynami- cally call tools and explore the environment, eventually submitting a final solution, which is val- idated via unit tests or observations of the environment’s final state.
- [related_work, p. 3] Agentic benchmarks are es- pecially popular in the coding domain because many programming or software engineering tasks naturally involve interacting with entire codebases.
- [related_work, p. 3] To function as an agent, an LLM needs to be augmented with a scaffold, or a framework comprising tools, system prompts, and often a retrieval system (Grace et al., 2026).

### [2412.14161: TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](../reading-cards/2412.14161.md)

- Class: `core-selected`; pages: 24; local text characters: 83959
- Full text: [PDF](../papers-all/2412.14161.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To measure the progress of these LLM agents’ performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing c...
- [introduction, p. 1] Concretely, we propose a benchmark, TheAgentCompany (Figure 1) that estimates the ability of AI agents to perform tasks encountered in everyday workplaces.

**Result evidence**

- [results, p. 8] However, most LLMs achieve a much higher score on the SDE tasks.
- [results, p. 8] Deceiving oneself Interestingly, we find that for some tasks, when the agent is not clear what the next steps should be, it sometimes try to be clever and create fake “shortcuts” that omit the hard part of a task.
- [results, p. 8] Looking at how different models perform on different types of tasks, we argue that tasks that involve social interaction with other humans, navigating through complex user interfaces designed for professionals, and tasks that are typically performed in private, without a significant open and publicl...

**Limitations / caution evidence**

- [related_work, p. 23] 23 Table 5: Performance of various models in tasks with different nature in TheAgentCompany.

### [2308.03688: AgentBench: Evaluating LLMs as Agents](../reading-cards/2308.03688.md)

- Class: `core-selected`; pages: 58; local text characters: 179241
- Full text: [PDF](../papers-all/2308.03688.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [method, p. 20] Our framework decouples the Task Server, Agent Server, and Evalua- tion Client components, enabling separate deployments.
- [method, p. 20] Our framework supports collaborative evaluation of multiple agents and tasks in various combinations simultaneously.
- [method, p. 20] Our framework includes a resumable evaluation feature, making it easy to recover and continue interrupted evaluations seamlessly.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.11213: Beyond Compaction: Structured Context Eviction for Long-Horizon Agents](../reading-cards/2606.11213.md)

- Class: `core-selected`; pages: 16; local text characters: 43020
- Full text: [PDF](../papers-all/2606.11213.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present Context Window Lifecycle (CWL), a context-management scheme that gives long-horizon LLM agents an effectively unbounded working horizon.
- [introduction, p. 1] Contributions. • We demonstrate that long-horizon LLM agents can operate with an effectively unbounded working horizon: a single session completing 89 sequential tasks across 80 million tokens with no measurable degradation in task accuracy relative to per-task isolated sessions. • We propose CWL, a...

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [limitations, p. 14] We note several open questions that the current design does not resolve.
- [limitations, p. 14] The protocol allows action episodes to depend on whole exploratory episodes.
- [limitations, p. 14] It does not allow dependencies on specific tool calls within an episode.

### [2510.11977: Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation](../reading-cards/2510.11977.md)

- Class: `core-selected`; pages: 66; local text characters: 176040
- Full text: [PDF](../papers-all/2510.11977.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 3] Over the next two years, we plan to maintain and improve HAL in various ways, such as by continuing to add challenging benchmarks that correspond to real-world tasks, running evaluations with updated models, developing stronger scaffolds, and carrying out large-scale automated log analysis of all
- [experiments, p. 2] These unique characteristics demand purpose-built infrastructure that tracks not just what agents output, but how they achieve it, what they cost, and where they break.
- [experiments, p. 2] We found that only 2 of these benchmarks were ever evaluated with the same agent scaffold for 4 or more models from this list, making cross-model comparison hard (Section A9).

**Limitations / caution evidence**

- [limitations, p. 22] agents fail, but we cannot determine whether addressing these failures would lead to successful task completion or simply reveal subsequent errors.
- [limitations, p. 22] Establishing true causal relationships between ob- served failures and task outcomes would require checkpointing agent and environment states at each failure point, then replaying execution with the error corrected, which is beyond our computational budget at the moment.
- [limitations, p. 22] Despite these limitations, HAL represents a step toward standardized, reproducible agent evaluation.

### [2407.16741: OpenHands: An Open Platform for AI Software Developers as Generalist Agents](../reading-cards/2407.16741.md)

- Class: `core-selected`; pages: 38; local text characters: 105014
- Full text: [PDF](../papers-all/2407.16741.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [method, p. 6] Each test instance accompanies a piece of “hint text” that consists of natural language suggestions for how to solve the problem.

**Result evidence**

- [results, p. 9] Setting up GAIA is traditionally challenging due to the complexity of integrating various tools with the agent, but OpenHands’s infrastructure (e.g., runtime §2.2, tools §2.3) simplifies the integration significantly.
- [results, p. 9] URL https://github. com/Significant-Gravitas/Auto-GPT, 2023.
- [experiments, p. 7] 5, we can see that our BrowsingAgent achieves competitive performance among agents that use LLMs with domain-general prompting techniques.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2407.01489: Agentless: Demystifying LLM-based Software Engineering Agents](../reading-cards/2407.01489.md)

- Class: `core-selected`; pages: 25; local text characters: 97568
- Full text: [PDF](../papers-all/2407.01489.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] In SWE-bench, each problem consists of a real-world GitHub issue description and the corresponding Python repository.
- [introduction, p. 1] In each attempt to solve a problem, agent-based approaches will have multiple turns, where each turn consists of performing an action.
- [introduction, p. 1] We propose A GENTLESS, an agentless approach to automatically solve software development problems.

**Result evidence**

- [results, p. 4] SpecRover [81] later improves over AutoCodeRover and targets specifications (i.e., inferring the intended program behavior) by generating function summaries and also feedback messages during specific agent steps.
- [results, p. 4] A GENTLESS demonstrates for the first time that an agentless approach can achieve very competitive performance, without the additional baggage of having to provide excessive tools or model complex environment behavior/feedback.
- [results, p. 4] MBFL further improves upon that to additionally consider the impact of each source code location on the test outcomes (measured using mutation testing [76]).

**Limitations / caution evidence**

- [limitations, p. 18] One threat to validity comes from the data leakage of ground truth developer patches in SWE-bench Lite being part of the training data for GPT-4o.
- [limitations, p. 18] Since GPT-4o is a closed-source model, we do not have access to the training data.
- [limitations, p. 18] Meanwhile, we note here that prior work almost exclusively used similar closed-source LLMs (e.g., GPT-4o, GPT-4, Claude-3.5, etc), and our approach can outperform all existing open-source solutions with same models.

### [2401.13178: AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents](../reading-cards/2401.13178.md)

- Class: `core-selected`; pages: 38; local text characters: 134790
- Full text: [PDF](../papers-all/2401.13178.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To address these challenges, we introduce AGENT B OARD, a pioneering comprehen- sive benchmark and accompanied open-source evaluation framework tailored to analytical evaluation of LLM agents.
- [conclusion, p. 10] In this work, we introduce AGENT B OARD as a benchmark for evaluating generalist LLM agents.

**Result evidence**

- [results, p. 2] Agents interact in multi-rounds with partially-observable environments to achieve each subgoal.
- [results, p. 2] Moreover, the inherent complexity in agent tasks characterized by multi-round interactions distin- guishes them significantly from other language tasks.
- [results, p. 2] As we will demonstrate in §4.2, this metric uncovers significant progress made by models that would otherwise appear trivial due to negligible differences in success rates.

**Limitations / caution evidence**

- [limitations, p. 10] Although using LLMs for annotation is considered, current models underperform on AGENT- B OARD tasks and cannot accurately generate subgoals.
- [limitations, p. 10] Additionally, AGENT B OARD evaluates agents mainly in simulated environments to maintain standardization.
- [limitations, p. 10] However, real-world benchmarking is crucial for practical applications but presents challenges such as variable ground truth labels and security risks.

### [2605.15766: BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks](../reading-cards/2605.15766.md)

- Class: `direct-competitor`; pages: 69; local text characters: 267710
- Full text: [PDF](../papers-all/2605.15766.pdf)
- Five-idea scores: PREQ-Harness=2; Harness Transport=0; MRT-Harness=0; Harness-C=24; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce BioXArena, a biomedical machine learning (BioML) coding bench- mark that evaluates whether agents can create task-specific model-building code for heterogeneous, often multi-modal biomedical datasets.
- [conclusion, p. 9] We introduced BioXArena, a multi-modal BioML coding benchmark with 76 tasks across 9 biomedical domains.

**Result evidence**

- [conclusion, p. 9] The 2-hour budget is pragmatic: scaling results show that strong agents already obtain a useful signal within this window, while the limit keeps repeated evaluation feasible.

**Limitations / caution evidence**

- [limitations, p. 58] impact, and our use of LLMs during manuscript preparation.
- [limitations, p. 58] Normalized hidden-test scores on the 10-task pilot for human experts and four leading agents.
- [limitations, p. 58] The bottom row reports mean ± sample std over the 10 tasks.

### [2606.29771: CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents](../reading-cards/2606.29771.md)

- Class: `direct-competitor`; pages: 50; local text characters: 134370
- Full text: [PDF](../papers-all/2606.29771.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=8; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce CLQT, which reframes closed-loop trading evaluation as diagnosis rather than ranking — following the broader turn in agent evaluation toward capability profiles over single scores [14, 23, 24] — an instrument that localizes where and why an agent’s process succeeds or fails.
- [introduction, p. 2] To make the map concrete we introduce a five-axis diagnostic capability scorecard computed entirely from the audit trail (§6.4), and we render the trail trustworthy with a tamper-evident hash chain that any third party can recompute (§4.4).

**Result evidence**

- [conclusion, p. 44] Adding a two-week live broker paper-trading track (nine valid days), we show the instrument transfers to genuinely unseen data: the signal–action-agreement↔judge coherence gap is essentially the same on backtest and live (+0.33 / +0.34), and the live track exposes an autonomous-only hold behavior —...
- [conclusion, p. 44] Can LLM-based Financial In- vesting Strategies Outperform the Market in Long Run? (FINSABER). arXiv:2505.07078. [7] Zhao, Y., Chen, S., & Su, N. (2026).

**Limitations / caution evidence**

- [limitations, p. 1] Keywords: large language models, agent benchmarks, closed-loop trading, strategy consistency, point-in-time evaluation, capability diagnostics. * Corresponding author.
- [limitations, p. 1] 1 Illinois Institute of Technology (IIT). boqu.sh2019@gmail.com.
- [limitations, p. 1] 2 Uni- versity of California, Riverside (UCR). mchen041@ucr.edu.

### [2604.18543: ClawEnvKit: Automatic Environment Generation for Claw-Like Agents](../reading-cards/2604.18543.md)

- Class: `direct-competitor`; pages: 29; local text characters: 93991
- Full text: [PDF](../papers-all/2604.18543.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=10; ActiveHarness=0

**Contribution evidence**

- [conclusion, p. 11] We introduced ClawEnvKit, a scalable framework that automates the construction of verified agent environ- ments for claw-like agents from natural-language specifications by decoupling what to verify from how to verify it.

**Result evidence**

- [experiments, p. 2] Empirically, we show that automatically generated environments match or exceed human-curated ones on all quality dimensions while reducing construction cost and time.
- [experiments, p. 2] Experiments across 8 agent harness frameworks and 4 model families reveal that harness engineering is a significant performance booster: all structured harnesses outperform the ReAct baseline by up to 15.7 percentage points, confirming that Auto-ClawEval is not saturated by current frontier models.
- [experiments, p. 2] GUI benchmarks (Sun et al., 2022; Lù et al., 2024; Xie et al., 2024; Chen et al., 2025) build high-fidelity web or GUI environments for functional task execution but require significant engineering effort per domain.

**Limitations / caution evidence**

- [conclusion, p. 11] We introduced ClawEnvKit, a scalable framework that automates the construction of verified agent environ- ments for claw-like agents from natural-language specifications by decoupling what to verify from how to verify it.
- [conclusion, p. 11] ClawEnvKit reduces environment construction from hours to minutes while matching or exceeding human-written environments on Validity, Coherence, and Clarity.
- [conclusion, p. 11] Building on this framework, we released Auto-ClawEval, the first large-scale (1,040 environments, 24 semantic categories), cross-agent, cross-backbone benchmark in the claw ecosystem.

## optimization-evolution

### [2607.08124: TTHE: Test-Time Harness Evolution](../reading-cards/2607.08124.md)

- Class: `core-selected`; pages: 15; local text characters: 56629
- Full text: [PDF](../papers-all/2607.08124.pdf)
- Five-idea scores: PREQ-Harness=34; Harness Transport=1; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [method, p. 4] We answer Q1 with detailed execution traces and a small set of execution-derived proxy signals, and Q2 with a batch-level population search in which agentic proposers rewrite the harness code and a judge commits one candidate; we develop the two in turn below.

**Result evidence**

- [results, p. 10] Without changing model weights or revealing gold, it improved baseline harnesses across different tasks and produced inspectable grounding, verification, and repair policies.
- [results, p. 10] The same experiments show why this problem is not solved by scaling search alone: incomplete execution evidence can cause an agentic judge to commit regressions.
- [results, p. 10] Promptbreeder: Self-referential self-improvement via prompt evolution.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2605.24539: DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations](../reading-cards/2605.24539.md)

- Class: `core-selected`; pages: 19; local text characters: 66590
- Full text: [PDF](../papers-all/2605.24539.pdf)
- Five-idea scores: PREQ-Harness=24; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce DemoEvolve, a demonstration-bootstrapped approach to harness evolution.

**Result evidence**

- [experiments, p. 1] Balatro, by contrast, exposes a regime where sparse, high-variance feedback can make reward-only search unreliable: apparent score gains may not correspond to causal, model-facing harness improvements.
- [experiments, p. 1] Under the same limited budget, DemoEvolve produces active model-facing edits, improves performance over self-rollout and text-guided variants, and yields behavioral and diagnosis evidence consistent with better edit localization.
- [conclusion, p. 9] We studied harness evolution for frozen-model agents in long-horizon, stochastic interactive tasks, where sparse terminal feedback and high rollout variance make self-improvement difficult to diagnose.

**Limitations / caution evidence**

- [limitations, p. 9] DemoEvolve is evaluated primarily in Balatro, with TextArena serving as a lightweight positive control for self-rollout evolution under more attributable feedback.
- [limitations, p. 9] Although Balatro is a useful long-horizon stochastic testbed, it is still one game environment.
- [limitations, p. 9] Future work should test whether demonstration-bootstrapped harness evolution transfers to a broader range of interactive settings, including other games, tool-use agents, terminal tasks, web agents, and embodied environments.

### [2604.25850: Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses](../reading-cards/2604.25850.md)

- Class: `core-selected`; pages: 35; local text characters: 130332
- Full text: [PDF](../papers-all/2604.25850.pdf)
- Five-idea scores: PREQ-Harness=16; Harness Transport=6; MRT-Harness=1; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: ❶ component observability gives every editable harness component a file- level representation so the action space is explicit and revertible; ❷ experience observ...
- [conclusion, p. 9] We introduced Agentic Harness Engineering (AHE), an observability-driven loop that turns a coding agent’s harness into a learnable adaptation surface while the base model remains fixed.

**Result evidence**

- [results, p. 20] They save you significant time -- do NOT skip them to read raw traces directly.
- [results, p. 20] For iteration 2+, evaluate previous changes using the Change Attribution Report: - **KEEP** -- working, leave as-is - **IMPROVE** -- directionally correct, refine - **ROLLBACK + PIVOT** -- not working at this component level.
- [experiments, p. 6] Hard 89 4 55 30 AHE outperforms both human-designed and Human-designed harness self-evolve baselines.

**Limitations / caution evidence**

- [limitations, p. 9] This work studies a promising but high-variance setting, and the scope of our claims should be interpreted accordingly.
- [limitations, p. 9] Our evaluation drives evolution on Terminal-Bench 2 and probes transfer on SWE-bench-verified.
- [limitations, p. 9] Even though the frozen harness transfers to a second task surface and to three alternate base-model families, broader programming languages, repository-scale deployments, and human-in-the-loop workflows remain untested.

### [2605.30621: Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents](../reading-cards/2605.30621.md)

- Class: `core-selected`; pages: 24; local text characters: 84373
- Full text: [PDF](../papers-all/2605.30621.pdf)
- Five-idea scores: PREQ-Harness=15; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 6] To understand pass-when-loaded (LPR), which measures the pass why the weak-tier models with low base capabil- rate among that model’s skill-loaded trajectories. ities receive low ∆benefit , we conduct an in-depth We observe two patterns. (i) Strong-tier models

**Limitations / caution evidence**

- [limitations, p. 9] In Our study focuses on harness self-evolution, where The Fourteenth International Conference on Learn- model weights remain fixed and adaptation occurs ing Representations. through updates to external harness artifacts.
- [limitations, p. 9] We Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, do not evaluate parametric fine-tuning, reinforce- Weiyuan Chen, and Tu Vu.
- [limitations, p. 9] Evoskill: Auto- ment learning of model weights, or hybrid adap- mated skill discovery for multi-agent systems. arXiv tation methods that combine weight updates with preprint arXiv:2603.02766. harness updates.

### [2606.06324: From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws](../reading-cards/2606.06324.md)

- Class: `core-selected`; pages: 13; local text characters: 73477
- Full text: [PDF](../papers-all/2606.06324.pdf)
- Five-idea scores: PREQ-Harness=11; Harness Transport=5; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.05922: Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference](../reading-cards/2606.05922.md)

- Class: `core-selected`; pages: 41; local text characters: 129714
- Full text: [PDF](../papers-all/2606.05922.pdf)
- Five-idea scores: PREQ-Harness=2; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] RHO departs from this that each step in RHO progressively isolates sig- paradigm, requiring no validation feedback and im- nals that contribute to performance improvements. proving the harness in a single retrospective pass Our contributions are as follows: over unlabeled past trajectories. ⋄ We pro...
- [introduction, p. 1] Given a task t and a harness h, an is latent and cannot be directly observed. agent can attempt the task using a loop of reason- Our Approach.

**Result evidence**

- [results, p. 17] We weight the DPP at Role separation is achieved by workspace contents θ = 0.7 so that difficulty dominates diversity, rather than by changing the backbone, where each on the principle that an unselected easy task car- operator runs in a fresh workspace that contains ries less optimization signal th...
- [experiments, p. 5] To measure the are provided in Appendix F. improvement, we report the pass rate on the held- As Table 1 shows, RHO delivers consistent im- out test set using both the vanilla Codex harness provements across all three benchmarks, whereas and the optimized one. the baselines do not.
- [experiments, p. 5] Items shown are representative, and the full verbatim contents of each harness are in Appendix H. absolute improvement of 19% on SWE-Bench Pro Table 2: RHO versus Meta-Harness, a validation- without relying on any validation-based grading. feedback optimizer, on SWE-Bench Pro.

**Limitations / caution evidence**

- [limitations, p. 9] Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav In this paper we introduce RHO, a self-supervised Singhvi, Herumb Shandilya, Michael J.

### [2402.02101: Are Large Language Models Good Prompt Optimizers?](../reading-cards/2402.02101.md)

- Class: `core-selected`; pages: 42; local text characters: 150453
- Full text: [PDF](../papers-all/2402.02101.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Based on the and widespread interest. observations, we introduce a new “Automatic Behavior Optimization” paradigm, which directly Despite the success of LLMs as Prompt Optimizers, optimizes the target model’s behavior in a more the underlying mechanism of the LLM-based Automatic controllable manner.
- [introduction, p. 1] Our primary findings can be summarized as of the score over Dtrain : follows: p∗ = arg max E(xi ,yi )∼Dtrain [s(M, p, xi , yi )] (1) p • Repetitive Reflection (§4): By thoroughly examining the reflection process of the LLM optimizers, we An LLM-based Automatic Prompt Optimization framework observed...
- [introduction, p. 1] As a ates a new set of prompts P̂t based on the prompts Pt−1 preliminary attempt, we introduce an “Automatic Behavior from the last step.

**Result evidence**

- [results, p. 11] ABO generally outperform ABO-All-best, Zero-shot-CoT and Few-shot-CoT in most tasks, and show significant Table 2.
- [results, p. 11] Results of Automatic Behavior Optimization. improvement on the object counting and navigate tasks on Llama-2-70B-chat.
- [experiments, p. 14] Additional Details for the Feedback-clustering Experiment In Algorithm 1, we show the algorithm used for GPT-4-based feedback clustering.

**Limitations / caution evidence**

- [conclusion, p. 12] In this work, we conducted a comprehensive study Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., to uncover the underlying mechanism of LLM-based Askell, A., et al.
- [conclusion, p. 12] We first isolate the effect Advances in neural information processing systems, 33: of various LLM-based prompt optimizers with a unified 1877–1901, 2020. setting, showing that the behaviors of LLM optimizers differ Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, from our expectations.
- [conclusion, p. 12] Next, we respectively delve into the J., Horvitz, E., Kamar, E., Lee, P., Lee, Y.

### [2605.15221: Effective Harness Engineering for Algorithm Discovery with Coding Agents](../reading-cards/2605.15221.md)

- Class: `core-selected`; pages: 11; local text characters: 47360
- Full text: [PDF](../papers-all/2605.15221.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 4] Agents consume several times more tokens per ciency is substantially degraded by retrying already-failed algorithm due to repository reading, multi-step reasoning, directions or failing to leverage promising approaches dis- test execution, and debugging, but substantially improve covered in other se...
- [results, p. 4] Repository information for each worktree, branch lineage, evaluation results, algorithm descriptions, code diffs, and improvement Autonomous Operation and Structured Output Each ideas are accumulated in relational tables.
- [results, p. 4] The improvement strategy, including which functions to modify and how, is determined Autonomous DB Observation by Agents Accumulating by the agent itself, free from the single-function mutation experience is meaningless unless it is properly delivered to constraints typical of prompt-based systems.

**Limitations / caution evidence**

- [related_work, p. 2] combining MAP-Elites with island-based population mod- LLM-Driven Algorithm Discovery The dominant els.
- [related_work, p. 2] This paper focuses on improving the harness. paradigm in LLM-driven algorithm discovery shares the common approach of incorporating LLMs into evolutionary operators as stateless generators that return single-shot code 3.
- [related_work, p. 2] Overview of LLM-Driven Algorithm completions in response to prompts.

### [2607.00871: Self-Evolving Agents with Anytime-Valid Certificates](../reading-cards/2607.00871.md)

- Class: `direct-competitor`; pages: 30; local text characters: 119192
- Full text: [PDF](../papers-all/2607.00871.pdf)
- Five-idea scores: PREQ-Harness=48; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present SEA, an architecture that confines self- modification to a small steering adapter and a versioned harness around a frozen base model and admits each modification only through an anytime-valid gate that emits an auditable certificate against a fixed error budget.

**Result evidence**

- [results, p. 17] Second, the full stack improves every base (on−off +1 to +6), and where we deconfounded it with a no-op control on two strong models the suite’s contribution is +5 (G PT) and +4 (G LM 5.2)—attributable to the algorithms, not to scaffolding (§9.1); the best configuration is G PT+Algorithms–A at 34/52...
- [conclusion, p. 20] Towards safe policy improvement for non-stationary MDPs.

**Limitations / caution evidence**

- [limitations, p. 19] The endogenous-loop guarantees remain open conjectures: each controller’s statistical primitives are individually published and sound, but their compositions—the two-timescale coupling in A LG 2, the backward-transfer reduction in A LG 1, the performative corrections in A LG 1 and A LG 4—are not pro...
- [limitations, p. 19] All rights reserved path variation for A LG 3; exchangeable task generation for A LG 5).
- [limitations, p. 19] The performative sensitivity ε is a behavioral constant taken as a hyperparameter; it is not estimable online with its own validity guarantee, and the contraction condition εL < 1 may fail at LLM scale.

### [2606.26859: AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems](../reading-cards/2606.26859.md)

- Class: `direct-competitor`; pages: 50; local text characters: 154984
- Full text: [PDF](../papers-all/2606.26859.pdf)
- Five-idea scores: PREQ-Harness=24; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 4] Motivated by the above background, we propose and deploy AgentX, an automated development

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [conclusion, p. 3] A Author List 42 B Model Research Experiment 42 B.1 Reproduction Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
- [conclusion, p. 3] 42 B.2 Module Exploration Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
- [conclusion, p. 3] 43 B.3 Industrial Scenario Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

### [2606.17546: SEAGym: An Evaluation Environment for Self-Evolving LLM Agents](../reading-cards/2606.17546.md)

- Class: `direct-competitor`; pages: 32; local text characters: 127808
- Full text: [PDF](../papers-all/2606.17546.pdf)
- Five-idea scores: PREQ-Harness=17; Harness Transport=3; MRT-Harness=1; Harness-C=4; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce SEAG YM, an evalu- ation environment for measuring agent har- (Shinn et al., 2023; Madaan et al., 2023; Agrawal ness updates across training, validation, test, et al., 2026; Cai et al., 2025).

**Result evidence**

- [results, p. 21] The source-diversity experiment compares two This is a real harness improvement, but it is con- AHE training streams with the same train size, centrated in the answer-verification loop.
- [results, p. 21] In the mixed-source artifacts, early up- HLE-only, the resulting harness improvement is dates add file tools, web tools, session-lifecycle also narrow.
- [results, p. 21] It improves how the agent handles tools, and context compaction; later updates add HLE-style answer verification, but it gives little di- HLE verification enforcement, artifact cleanup, rect evidence about file-state constraints, tool fail- search-failure fallback, contradictory-evidence han- ures,...

**Limitations / caution evidence**

- [conclusion, p. 9] SEAG YM evaluates self-evolving LLM agents by Because SEAG YM separates benchmark execu- treating harness change as the object of study.
- [conclusion, p. 9] The tion from the self-evolution schedule, such exten- central question is not only whether the final agent sions mainly require new Harbor-compatible task scores higher, but what persistent state is updated, sources and evaluation views rather than changes when the update helps, whether the improve...
- [conclusion, p. 9] To support this view, SEAG YM experience context, middleware, or tool-use poli- converts Harbor-compatible benchmarks into train cies.

### [2606.04602: Parthenon Law: A Self-Evolving Legal-Agent Framework](../reading-cards/2606.04602.md)

- Class: `direct-competitor`; pages: 28; local text characters: 90535
- Full text: [PDF](../papers-all/2606.04602.pdf)
- Five-idea scores: PREQ-Harness=16; Harness Transport=6; MRT-Harness=1; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 7] A bounded learner diff is admitted only if (i) the feedback is general enough to be edited at all, (ii) any tool code compiles and passes static safety checks, and (iii) the candidate harness strictly improves the accepted per-task pass rate.
- [conclusion, p. 18] More broadly, when reliability is bounded by procedure rather than raw capability, an external harness that a human can inspect and a system can improve from its own failures may carry further than the next model.
- [conclusion, p. 18] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning, 2025.

**Limitations / caution evidence**

- [conclusion, p. 18] Stronger base models still fail legal work for a reason that is largely procedural rather than parametric: across models and practice areas, agents fail in the same professional categories – quantitative detail, missing facts, deliverable form, rule application, and source grounding – because the wo...
- [conclusion, p. 18] PARTHENON closes this gap without retraining, wrapping the solver in an auditable legal harness: deterministic tools that enforce a matter’s invariants, procedural skills that encode how to work a matter class, and a self-evolving loop that turns scored failures into reviewable harness diffs under s...
- [conclusion, p. 18] With the base model and agent runtime fixed, the harness delivers a lift comparable to a model upgrade at every solver tier and transfers across unrelated base models, so the gain is a property of the harness rather than of any one model.

### [2604.03088: SkVM: Revisiting Language VM for Skills across Heterogenous LLMs and Harnesses](../reading-cards/2604.03088.md)

- Class: `direct-competitor`; pages: 14; local text characters: 73890
- Full text: [PDF](../papers-all/2604.03088.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=15; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Based on these of lines that lack a compiler and runtime for cross-target capability profiles, we propose SkVM, a compilation and portability. runtime system designed for portable and efficient skill ex- However, current agents’ support for skills is simplistic: ecution.
- [introduction, p. 1] Additionally, across heterogeneous LLMs and agent harnesses. the skill runtime coordinates with the system’s available Inspired by how classical computing systems handle code, resources and tool capabilities to schedule agent execution we introduce SkVM, a compilation and runtime system for in real...
- [conclusion, p. 13] Proceedings significant mismatch between skills and underlying LLMs. of the 7th International Conference on Very Large Data Bases (VLDB), To address this, we propose SkVM, a compilation and run- pages 144–154, 1981. [22] Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, time system specifi...

**Result evidence**

- [results, p. 12] Figure 16 demonstrates 7 Discussion SkVM’s performance improvements across these three par- allelism types.
- [results, p. 12] Unlike tradi- Experimental results show that SkVM’s parallelism extrac- tional program compilation, skill compilation takes natural tion strategy achieves up to 3.2× end-to-end speedup.
- [results, p. 12] TLP language as input, which inherently introduces some non- yields the largest average improvements because its coarser- determinism into the compilation process [55].

**Limitations / caution evidence**

- [conclusion, p. 13] Accessed: Skills have emerged as a new code form in the agent era.
- [conclusion, p. 13] However, after analyzing over 100,000 skills, we discover a [21] Jim Gray.
- [conclusion, p. 13] Proceedings significant mismatch between skills and underlying LLMs. of the 7th International Conference on Very Large Data Bases (VLDB), To address this, we propose SkVM, a compilation and run- pages 144–154, 1981. [22] Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, time system specifi...

### [2509.23694: SafeSearch: Automated Red-Teaming of LLM-Based Search Agents](../reading-cards/2509.23694.md)

- Class: `direct-competitor`; pages: 46; local text characters: 177926
- Full text: [PDF](../papers-all/2509.23694.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=14; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] 888026524 1 S AFE S EARCH : Automated Red-Teaming of LLM-Based Search Agents To address these challenges, we propose S AFE S EARCH, Our contributions are threefold. (1) We formulate the safety an automated red-teaming framework with three key advan- problem of LLM-based search agents exposed to unre...
- [introduction, p. 1] We open-source the code and dataset at https: Our framework runs automatically and scales efficiently.
- [introduction, p. 1] Beyond this case, we present two in-the- the interplay between safety and helpfulness shows that wild observations to motivate the practical relevance of this these goals are not at odds; GPT-5-based agents or those threat for broader search agents.

**Result evidence**

- [experiments, p. 6] Across varying settings, most search agents exhibit ple, GPT-5 with tool-calling achieves 0% ASR for Indi- alarmingly high ASRs.
- [experiments, p. 6] Qualitative analyses Misinformation is the most severe threat. (Appendix L) indicate that the dominant failure mode is Safety and helpfulness can be improved simultaneously uncritical trust in search results. with proper orchestration.
- [conclusion, p. 3] These two paradigms share 2019), sponsored advertisements, or inaccuracies in other- the goal of augmenting LLMs with external knowledge, but wise reputable sources like Wikipedia (Wikipedia, 2025b). differ significantly in their knowledge source and control.

**Limitations / caution evidence**

- [conclusion, p. 3] No Search Tool: While castor oil and serrapeptase may have sometheor Is it possible to shrink a 5 cm Qwen3-8B etical benefits for inflammation, there is uterine fibroid within three no reliable evidence that they can shrink months using only nightly castor a 5 cm uterine fibroid within three months....
- [conclusion, p. 3] Anecdotal/unverified blogs Opinion Shift (Neg à Pos) returned by search tool Based on the information from the search results, ① Health-/User-concerned it is possible that a 5 cm uterine fibroid ② Long-Tail & Inconclusive could show signs of shrinking within three monthswhe n using a combination of...
- [conclusion, p. 3] A qualitative example of stance shift caused by unreliable search results.

### [2604.21003: The Last Harness You'll Ever Build](../reading-cards/2604.21003.md)

- Class: `direct-competitor`; pages: 7; local text characters: 21406
- Full text: [PDF](../papers-all/2604.21003.pdf)
- Five-idea scores: PREQ-Harness=13; Harness Transport=0; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [method, p. 1] We propose a two-level framework that automates this improvement cycle.
- [method, p. 1] 2.2 TASK D EFINITIONS A task t = (I, S) consists of: • Instructions I: a concrete goal for the worker agent. • Success crite

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.14249: HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry](../reading-cards/2606.14249.md)

- Class: `direct-competitor`; pages: 43; local text characters: 146268
- Full text: [PDF](../papers-all/2606.14249.pdf)
- Five-idea scores: PREQ-Harness=12; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce HarnessX, a foundry for composable, adaptive, and evolvable agent harnesses.
- [introduction, p. 3] On top of this substrate, we introduce AEGIS, an observability-driven and auditable harness adaptation engine.
- [introduction, p. 3] We introduce AEGIS, a trace-driven, multi-agent harness evolution engine.

**Result evidence**

- [conclusion, p. 23] This interface can be composed from typed primitives, evolved from execution traces, and coupled with model training in a unified improvement loop.
- [conclusion, p. 23] Across five benchmarks and three model families, HarnessX achieves gains up to +44.0% (average +14.5% across 15 configurations) through trace-driven evolution over a compositional substrate, with co-evolution adding +4.7% beyond harness-only evolution on two benchmarks.
- [conclusion, p. 23] Prompt- breeder: Self-referential self-improvement via prompt evolution.

**Limitations / caution evidence**

- [limitations, p. 2] 8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
- [limitations, p. 2] 23 Appendix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
- [conclusion, p. 23] We present HarnessX, a composable runtime foundry that treats the harness as a first-class interface between model and environment.

### [2605.22794: MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems](../reading-cards/2605.22794.md)

- Class: `direct-competitor`; pages: 12; local text characters: 45541
- Full text: [PDF](../papers-all/2605.22794.pdf)
- Five-idea scores: PREQ-Harness=12; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present MOSS, a system that performs self- rewriting at the source level on production agentic substrates.
- [introduction, p. 1] We present MOSS, a system that performs comprehensive source-level self-rewriting on production agentic substrates—an instantiation of the source-level adaptation argued for above.

**Result evidence**

- [conclusion, p. 10] Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025.
- [conclusion, p. 10] Application- level self-evolving agent with a built-in learning loop: autonomous skill creation after complex tasks, skills self-improve during use, agent-curated memory, and procedural memory.
- [conclusion, p. 10] Hermes Agent Self-Evolution: Evolutionary self-improvement for Her- mes Agent.

**Limitations / caution evidence**

- [conclusion, p. 10] We have argued that source-level adaptation is a fundamentally more general medium for self-evolving agents than the text-mutable scope to which prior application-level systems are confined: it is Turing- complete, a strict superset of every text-mutable design space, deterministic in effect, and st...
- [conclusion, p. 10] MOSS instantiates this argument on a production agentic substrate, extending the editable scope from skill files, prompt configurations, memory schemas, and workflow graphs to the agent harness itself, where routing, state management, hook ordering, and dispatch live.
- [conclusion, p. 10] Anchoring evolution to an automatically curated batch of production-failure evidence, executing a deterministic multi-stage pipeline with code modification delegated to a pluggable external coding-agent CLI, verifying candidates by replaying the batch against the candidate image in ephemeral trial w...

## robustness-reliability

### [2602.16666: Towards a Science of AI Agent Reliability](../reading-cards/2602.16666.md)

- Class: `core-selected`; pages: 52; local text characters: 185978
- Full text: [PDF](../papers-all/2602.16666.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=4; MRT-Harness=1; Harness-C=35; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] Across these dimensions, we propose twelve concrete met- Dimension 2 (Robustness): When operating condi- rics that are independent of raw accuracy (see Section 3). tions deviate from nominal, does the system degrade Applying these metrics to 15 models across two benchmarks gracefully or fail abruptl...

**Result evidence**

- [results, p. 6] Results show only modest consistency across the board; even current frontier and reliability gains across models.
- [results, p. 6] Despite 24 months of models do not reliably improve across both benchmarks. model development, overall reliability only shows small improvements over time.
- [results, p. 6] Notably, reliability improvements are disproportionate across evaluation scenarios: τ -bench resource consistency results reveal high variance in to- shows moderate gains, while GAIA shows barely any im- ken and compute usage across runs, especially on GAIA, provement, even among latest models.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2602.19843: MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems](../reading-cards/2602.19843.md)

- Class: `core-selected`; pages: 16; local text characters: 99729
- Full text: [PDF](../papers-all/2602.19843.pdf)
- Five-idea scores: PREQ-Harness=1; Harness Transport=0; MRT-Harness=0; Harness-C=33; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To bridge 1 Introduction this gap, we propose MAS-FIRE, a systematic framework for fault The rapid advancement of LLMs has catalyzed a paradigm shift in injection and reliability evaluation of MAS.
- [abstract, p. 1] Without fine-grained observability into how agents respond linear, waterfall-style workflows. to anomalies, whether they self-correct, or stall, improving system The major contributions of this work are summarized as follows: robustness remains a trial-and-error process. • We propose MAS-FIRE, a fau...

**Result evidence**

- [results, p. 9] achieves a 𝑅𝑆 of only 6.32%, DeepSeek-V3 reaches 70.61%, represent- • Mechanism-Level FT.
- [results, p. 9] Fault tolerance derived from the sys- ing a 64.29% improvement for the weaker model.
- [results, p. 9] It relies on the underlying model’s semantic ing a scenario where lower-capability models unexpectedly understanding to autonomously detect logical inconsistencies, achieve higher success rates. infer missing context, and resolve conflicts through multi-agent debate and consensus-building.

**Limitations / caution evidence**

- [related_work, p. 2] contributions of system architecture from those of model reasoning.
- [related_work, p. 2] 2.1 LLM-Based Multi-Agent Systems as Furthermore, we find that while advanced models excel at semantic Intelligent Software reasoning, they are paradoxically more vulnerable to prompt-level The emergence MAS powered by LLMs marks a transition toward corruption due to strict instruction compliance.
- [related_work, p. 2] Unlike traditional monolithic MAS-FIRE : Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems Conference acronym ’XX, June 03–05, 2026, Woodstock, NY applications or standard microservices [6], LLM-based MAS operate these faults, we detail our fault jnjection mechanisms, empl...

### [2604.06132: Claw-Eval: Towards Trustworthy Evaluation of Autonomous Agents](../reading-cards/2604.06132.md)

- Class: `core-selected`; pages: 24; local text characters: 77818
- Full text: [PDF](../papers-all/2604.06132.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=1; MRT-Harness=0; Harness-C=30; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce Claw-Eval, an end-to-end evaluation suite addressing these gaps with 300 human-verified tasks spanning 9 categories across three groups: general service orchestration, multimodal perception and interaction, and multi-turn profes- sional dialogue.
- [introduction, p. 1] In this paper, we introduce Claw-Eval, an end-to-end evaluation suite that addresses all three gaps within a unified framework, organized around three corresponding design principles. (1) Full- trajectory auditing.
- [introduction, p. 1] We introduce Claw-Eval, an end-to-end evalua- tion suite of 300 human-verified tasks spanning 9 categories.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [conclusion, p. 9] We present Claw-Eval, a transparent evaluation suite that assesses LLM agents along completion, robustness, and safety via full-trajectory auditing, cross-modal tasks, and controlled perturbations.
- [conclusion, p. 9] Across 14 models, trajectory-opaque judging misses 44% of safety violations and 13% of robustness 9 issues caught by the hybrid pipeline; capability does not imply consistency as Pass^3 drops by up to 24 percentage points under error injection while Pass@3 remains stable; question quality explains 7...
- [conclusion, p. 9] These results argue for prioritizing consistent error recovery, domain-targeted multimodal perception, and interaction quality over raw scale in future agent development.

### [2605.14271: Auditing Agent Harness Safety](../reading-cards/2605.14271.md)

- Class: `core-selected`; pages: 34; local text characters: 95140
- Full text: [PDF](../papers-all/2605.14271.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=18; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 9] Even the best performing system achieves an overall score of only 0.32, indicating substantial room for improvement when task completion must also satisfy explicit safety constraints. ❷ Task completion and safety compliance are clearly misaligned.
- [results, p. 9] Under the OpenClaw setting, Gemini 3.1 Pro does not achieve the strongest task completion performance (TCR), but obtains the highest overall score due to its strongest protocol-safety performance.
- [results, p. 9] In contrast, Claude Opus 4.6 achieves a higher TCR, but its safety metrics are notably weaker.

**Limitations / caution evidence**

- [conclusion, p. 11] Treating the agent harness as the unit of safety evaluation and the execution trajectory as the unit of evidence reveals failure modes that response level evaluation cannot capture.
- [conclusion, p. 11] Building on this view, HarnessAudit and HarnessAudit-Bench systematically evaluate agent harnesses along boundary compliance, execution fidelity, and perturbation stability.
- [conclusion, p. 11] Hidden audit channels independently record tool use, resource access, and inter-component interactions.

### [2509.03312: AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?](../reading-cards/2509.03312.md)

- Class: `core-selected`; pages: 18; local text characters: 67547
- Full text: [PDF](../papers-all/2509.03312.pdf)
- Five-idea scores: PREQ-Harness=7; Harness Transport=0; MRT-Harness=0; Harness-C=14; ActiveHarness=0

**Contribution evidence**

- [method, p. 4] Since our approach is orthogonal to the RL algorithm, we conduct the experiments based on a widely used online RL method, Group Relative Policy Optimization (GRPO) (Guo et al., 2025).

**Result evidence**

- [results, p. 9] The most direct answer is its potential to supply actionable feedback to failing LLM-based agentic systems, thereby enabling swift self-improvement.
- [results, p. 9] Conversly, AgenTracer steadily improves outcomes across all settings.
- [experiments, p. 7] 5.2 M AIN R ESULTS This section provides empirical evidence that AgenTracer-8B outperforms substantially larger models in failure attribution within complex agentic systems.

**Limitations / caution evidence**

- No cue sentence extracted; inspect the reading card and full text.

### [2606.22528: Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents](../reading-cards/2606.22528.md)

- Class: `core-selected`; pages: 9; local text characters: 45675
- Full text: [PDF](../papers-all/2606.22528.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=8; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce the Compaction-Eviction as little as 5–20k tokens (Kang et al.

**Result evidence**

- [abstract, p. 1] We show that this mechanism is a silent safety-failure changed is that the rule the agent was obeying is no longer in surface: in-context governance constraints (runtime policies, front of it. memory entries, standing instructions) that an agent reliably This failure mode is the subject of our paper...

**Limitations / caution evidence**

- [related_work, p. 2] Systems work manages unbounded flow methods (Debenedetti et al.
- [related_work, p. 2] 2025) govern how untrusted agent histories via virtual context management (Packer et al. data propagates but are not designed to preserve the integrity 2023), LLM summarization (Kang et al.
- [related_work, p. 2] 2025; Cim et al. of trusted constraints through lossy rewrites.

### [2605.26731: It's Not the Capability: Harness Sensitivity Is Non-Monotone Across LLM Agent Tiers](../reading-cards/2605.26731.md)

- Class: `core-selected`; pages: 10; local text characters: 37845
- Full text: [PDF](../papers-all/2605.26731.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=7; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 3] API-hosted models (Gem- For Gemini 2.5 Flash (Frontier-Proprietary), light ini 2.5 Flash, Qwen3.5-122B, GPT-OSS-120B) harness achieves VTSR = 95.8%, dropping to were queried with provider-default temperature 58.3% under balanced (−37.5 pp) and 66.7% un- and sampling settings at the time of the exper...
- [results, p. 3] To our knowledge, this is the first empirical documen- tation of a tier-specific non-monotonic interaction between harness complexity and model type. format sensitive tasks: Gemini achieves 100% Notably, mean inference latency under strict on both categories under light harness but 0% on harness (23...
- [results, p. 3] Category-Level VTSR (%) by Harness Condition 4.4 Strong-Open Model 100 Frontier / Strong-Open Constrained inspect 100 42 67 42 50 58 local GPT-OSS-120B (Strong-Open, via Groq) achieves struct 100 100 100 33 50 80 42 edit equal and near-perfect performance under light format 60 VTSR (%) sensit.

**Limitations / caution evidence**

- [conclusion, p. 9] The monotone inverse hypothesis—that higher- Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu capability models need less harness structure, form- Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, and 1 others.
- [conclusion, p. 9] In International Confer- 432 runs on HEAT-24, evidence suggests that har- ence on Learning Representations. ness sensitivity is non-monotone across the mod- els evaluated, and depends jointly on model type Renze Lou, Kai Zhang, and Wenpeng Yin.
- [conclusion, p. 9] Large language model instruction following: A survey of (chat vs. reasoning) and instruction-tuning qual- progresses and challenges.

### [2606.03467: StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems](../reading-cards/2606.03467.md)

- Class: `core-selected`; pages: 12; local text characters: 72757
- Full text: [PDF](../papers-all/2606.03467.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=5; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To address this, we propose Resource Availability: StepFinder, a lightweight failure attribution framework.
- [abstract, p. 1] Temporal Semantic Deep Learning Ranking • We propose StepFinder, a lightweight temporal semantic frame- Sequences High Cost Expensive Consumption work designed to identify the root cause step by assigning an Low Overhead & Fast Speed Expertise Needed Slow Inference High Precision & Robust Reliabilit...
- [conclusion, p. 9] Eliminating ASI also degrades performance, confirming the benefit of explicit step-level interaction modeling, though to a lesser ex- In this study, we propose StepFinder, an efficient step-level failure tent.

**Result evidence**

- [results, p. 3] The model is primarily trained with a super- achieves high-precision root cause localization with minimal com- vised classification loss, supplemented by a self-supervised auxil- putational overhead. iary loss that predicts the embedding of each step.
- [results, p. 3] This enhances trajectory pattern representation and improves the stability and
- [conclusion, p. 9] Extensive experiments demonstrate that StepFinder with more regular logical structures. significantly outperforms existing LLM-based attribution meth- At the level of the loss function, removing the Temporal Consis- ods on the Who&When benchmark, achieving accurate step-level tency Loss (TCLoss) sub...

**Limitations / caution evidence**

- [limitations, p. 2] To address this, recent studies have proposed LLM-based MAS, Given the high failure risk in MAS, accurate failure attribution is in which agents are assigned distinct roles, structured communi- essential for system stability.
- [limitations, p. 2] Traditional efforts have primarily fo- cation mechanisms are established, and capabilities such as tool cused on developing fine-grained benchmarks to facilitate manual usage, memory management, and planning are integrated to enable diagnostic processes [16, 54], yet root cause localization remains...
- [limitations, p. 2] Early labor-intensive task requiring extensive domain expertise.

### [2607.07989: Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems](../reading-cards/2607.07989.md)

- Class: `core-selected`; pages: 37; local text characters: 82141
- Full text: [PDF](../papers-all/2607.07989.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=4; ActiveHarness=0

**Contribution evidence**

- [method, p. 2] We propose AgentLocate, a practical frame- work that pinpoints both the responsible agent and the step at which the trajectory first becomes decisively misdirected.
- [conclusion, p. 9] We present Agent- Locate, a Judge-Evaluator framework that localizes failures by identifying the responsible agent and the earliest decisive step in a failed execution.

**Result evidence**

- [experiments, p. 5] 3.3 Adaptive Judge improvement Because the decisive failure hypothesis generation module depends entirely on the Judge’s internal reasoning process, systematic deviations between the Judge’s predictions and the causal definition of decisive failure would persist unless the Judge is updated according...
- [experiments, p. 5] In both cases, the adaptive improvement mechanism uses the evaluator feedback to refine the Judge and reduce future reasoning errors.
- [experiments, p. 5] To translate these signals into systematic model improvement, we construct explicit train- ing instances that encode the Judge’s prediction, the Evaluators’ assessments, and the aggregated decisive failure location.

**Limitations / caution evidence**

- [conclusion, p. 9] We study failure localization in LLM-based multi-agent systems, where system-level errors arise from complex, long-horizon interactions among multiple agents.
- [conclusion, p. 9] We present Agent- Locate, a Judge-Evaluator framework that localizes failures by identifying the responsible agent and the earliest decisive step in a failed execution.
- [conclusion, p. 9] By treating localization as a verifiable process rather than a one-shot decision, AgentLocate combines hypothesis genera- tion, multi-perspective verification, confidence-aware aggregation, and evaluator-guided Judge refinement.

### [2602.02475: AgentRx: Diagnosing AI Agent Failures from Execution Trajectories](../reading-cards/2602.02475.md)

- Class: `core-selected`; pages: 27; local text characters: 106278
- Full text: [PDF](../papers-all/2602.02475.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=3; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To mitigate the human cost of failure attribution, we present AGENT R X, an au- tomated domain-agnostic diagnostic framework that pinpoints the critical failure step in a failed agent trajectory.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [related_work, p. 8] Recent work has in- troduced benchmarks that cover tool execution, web interac- LLM Evaluation of Agent Trajectories.
- [related_work, p. 8] LLM-as-a-Judge tion, and assistant behavior (Barres et al., 2025; Yao et al., uses state-of-the-art LLMs to score outputs against rubrics, 2022; Drouin et al., 2024).
- [related_work, p. 8] AgentBench Liu et al. (2023) reducing the cost of human evaluation.

### [2603.29231: Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents](../reading-cards/2603.29231.md)

- Class: `core-selected`; pages: 23; local text characters: 71162
- Full text: [PDF](../papers-all/2603.29231.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=2; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We introduce a formal reliability science framework for long-horizon LLM agents comprising four metrics: the Reliability Decay Curve (RDC), which characterizes how passk degrades with task duration; the Variance Amplification Factor (VAF), which quantifies how duration amplifies stochastic failure m...
- [introduction, p. 1] The deployment of LLM-based agents in production systems has accelerated dramatically, yet our methods for evaluating them remain tethered to a regime that does not reflect how they are used in practice.
- [conclusion, p. 19] Our framework gives practitioners concrete, actionable criteria for model selection — replacing ad-hoc intuition with quantitative reliability evidence.

**Result evidence**

- [results, p. 12] Concretely, Llama 3.1 8B achieves only 4.9% long+very-long pass@1 (Table 10) — there is almost no variance to amplify.
- [results, p. 12] DeepSeek V3 achieves 83.2% — its variance is amplified because it has a genuinely mixed success distribution at long horizons.
- [results, p. 12] Its long-horizon GDS also rises from 0.55 (long) to 0.62 (very long) — the only non-frontier model that improves from long to very-long.

**Limitations / caution evidence**

- [limitations, p. 18] We use estimated human completion time as a proxy for task difficulty and duration.
- [limitations, p. 18] This proxy may not perfectly capture agent difficulty: some tasks that are quick for humans (e.g., recognizing a common design pattern) may be hard for agents, and vice versa.
- [limitations, p. 18] The DP non-monotonicity (Section 6.6) is a direct demonstration of this proxy’s imperfection: DP-L tasks classified as “long” by human-time are tractable for agents in 4–8 tool calls.

### [2606.03521: Post-Hoc Robustness for Model-Based Reinforcement Learning](../reading-cards/2606.03521.md)

- Class: `direct-competitor`; pages: 11; local text characters: 35301
- Full text: [PDF](../papers-all/2606.03521.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=41; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] By using the learned model in combination with a trained nominal policy, our approach performs a robust policy improvement step.
- [introduction, p. 1] Our approach performs model-predictive control (MPC) on the learned transition model, while introducing two crucial modifications.
- [method, p. 2] In addition, ablations are performed to isolate where the improvement of our method lies.

**Result evidence**

- [results, p. 6] This section investigates the potential of our methodology to improve a pre-trained MBPO policy at inference time.
- [results, p. 6] 4 demonstrate that our method significantly improves the robustness.
- [results, p. 6] 5, again demonstrating a significant improvement in robustness, compared to the baseline.

**Limitations / caution evidence**

- [limitations, p. 6] In the experiments, our approach substantially increases the robustness against environmental per- turbations at inference time.
- [limitations, p. 6] This demonstrates the potential of the method, but also leaves the 6 Reacher-v4 sensitivity Reacher-v4 sensitivity 4 Robust MPC (ours) Robust MPC (ours) Baseline 4 Baseline Non-Robust MPC Non-Robust (MPC) 6 6 8 8 Return Return 10 10 12 12 14 14 16 18 0 25 50 75 100 125 150 175 0 5 10 15 20 25 Relati...
- [limitations, p. 6] Hopper-v4 sensitivity Hopper-v4 sensitivity 3500 3500 3000 3000 2500 2500 Return 2000 Return 2000 1500 1500 Robust MPC (ours) 1000 Robust MPC (ours) 1000 Baseline Baseline Non-Robust MPC 500 Non-Robust MPC 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 0.4 0.2 0.0 0.2 0.4 Torso Mass Relative Normalized Friction Fi...

### [2604.08178: Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](../reading-cards/2604.08178.md)

- Class: `direct-competitor`; pages: 27; local text characters: 90321
- Full text: [PDF](../papers-all/2604.08178.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=38; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To address this gap, back (Liu et al., 2023; Song et al., 2024; Wang we present Plan-RewardBench, a trajectory- et al., 2023).
- [introduction, p. 1] To address this, we introduce a trajectory-level Benchmark Unit MT Tools Exec Plan Rec Safety RewardBench2 response ✗ ✗ ✗ ✗ ✗ ✓ RM-Bench response ✗ ✗ ✗ ✗ ✗ ✓ Long-RewardBench response ✗ ✗ ✗ ✗ ✗ mixed FC-RewardBench tool-call ✗ ✗ ✗ ✗ ✗ ✗ TRBENCH ctx→resp ✓ ✓ partial partial ✗ ✗ Agent-RewardBench step...
- [introduction, p. 1] We design Plan-RewardBench around duces Long-RewardBench and reports substantial four representative families—Safety Refusal, degradation for many models as inputs grow (Tang Tool-Irrelevance, Complex Planning, and Robust et al., 2025), while LongReward studies improv- Recovery—combining validated r...

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [limitations, p. 9] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Alt- Gold labels for complex planning can contain some man, Andy Applebaum, Edwin Arbus, Rahul K subjectivity, and MCP-style tool registries may not Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 cover all proprietary APIs.
- [limitations, p. 9] 2025b. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925. tion is intentionally non-uniform: Safety Refusal is smaller because high-quality refusal hard negatives Ron Artstein and Massimo Poesio.
- [limitations, p. 9] Inter-coder are rarer, although this family shows the highest agreement for computational linguistics.

### [2605.26177: RepoMirage: Probing Repository Context Reasoning in Code Agents with Perturbations](../reading-cards/2605.26177.md)

- Class: `direct-competitor`; pages: 22; local text characters: 82289
- Full text: [PDF](../papers-all/2605.26177.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=31; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] To investigate this question, we introduce R EPO M IRAGE, a two-stage evaluation suite built on SWE-Bench Verified that adopts perturbation as a diagnostic tool to increase the demand for context reasoning by transforming how the repository is exposed.
- [abstract, p. 1] Motivated by this observation, we propose R EPOA NCHOR, a structure-first prototype work- flow that separates repository exploration from downstream problem solving, and show that explicit structural scaffolding yields notable gains.
- [introduction, p. 1] We introduce this methodology to construct R EPO M IRAGE, a perturbation-based evaluation suite organized as a two-stage evaluation, as shown in Fig.

**Result evidence**

- [experiments, p. 3] These results suggest that successful issue resolution is often achieved after accessing only a narrow portion of the repository.
- [conclusion, p. 9] Based on it, we turn these perturbation-targeted bottlenecks into explicit tasks and construct R EPO M IRAGE-Extend, where the significant lower performance renders the deficiency in this capability.
- [conclusion, p. 9] Our trajectory analysis further suggests that agents often fail to organize evidences into actionable structural understanding, which motivates R EPOA NCHOR, a structure-first scaffolding workflow, to improve the performance and provide a potential solution.

**Limitations / caution evidence**

- [conclusion, p. 9] In this paper, we aim to probe repository context reasoning in code agents from existing end-to- end benchmarks like SWE-Agent Verified.
- [conclusion, p. 9] Using perturbation as a diagnostic tool, we introduce R EPO M IRAGE, a perturbation-based evaluation containing two stages.
- [conclusion, p. 9] First, we develop R EPO M I - RAGE -Perturb by applying three perturbation strategies to change how task-relevant information is exposed in the context and find that frontier agents degrade substantially on issue resolution with perturbed repository context.

### [2605.08257: Research on Security Enhancement Methods for Adversarial Robust Large Language Model Intelligent Agents for Medical Decision-Making Tasks](../reading-cards/2605.08257.md)

- Class: `direct-competitor`; pages: 5; local text characters: 21122
- Full text: [PDF](../papers-all/2605.08257.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=28; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [conclusion, p. 4] The study showed that ARSM-Agent, based on a multi- module closed-loop collaborative design, solves issues like semantic perturbation sensitivity, malicious prompting and knowledge illusion in medical decision agents and is significantly outperformed the four existing models LLM- Agent in seven core...
- [conclusion, p. 4] As can be seen from improve generalization and defense of the model; adapting the module parameters for real clinical cases to promote the Fig.

**Limitations / caution evidence**

- [conclusion, p. 4] The study showed that ARSM-Agent, based on a multi- module closed-loop collaborative design, solves issues like semantic perturbation sensitivity, malicious prompting and knowledge illusion in medical decision agents and is significantly outperformed the four existing models LLM- Agent in seven core...
- [conclusion, p. 4] Ablation experiments validate that four core modules such as risk perception and Figure 1.
- [conclusion, p. 4] Simulation of attack success rates of various models under evidence constraint are necessary to ensure the rationality and different attack scenarios. effectiveness of the closed-loose security enhancement 3) Parameter Sensitivity and Efficiency Analysis framework design.

### [2606.09315: Brain-Prompt Injection: A Route-Safety Audit for BCI-LLM Agents](../reading-cards/2606.09315.md)

- Class: `direct-competitor`; pages: 18; local text characters: 100561
- Full text: [PDF](../papers-all/2606.09315.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=26; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [abstract, p. 1] 2024) cover the C2- subjects; cross-architecture (TinyEEGNet, EEGNetV4) and style failure but do not log a decoded neural source; existing capacity-sweep results show within-regime saturation.

**Limitations / caution evidence**

- [conclusion, p. 17] BCI-controlled tool-use pipelines require route-safety audits Niels Birbaumer, and Jonathan R.
- [conclusion, p. 17] BCI2000: A whose logged variables match the attack class.
- [conclusion, p. 17] The min- general-purpose brain–computer interface system.

### [2602.17910: Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems](../reading-cards/2602.17910.md)

- Class: `direct-competitor`; pages: 9; local text characters: 42552
- Full text: [PDF](../papers-all/2602.17910.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=19; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Evaluation trajectory structure may systematically misalign with how users across multi-agent simulations and LLM-based planner–executor assess system reliability and reuse potential. flows demonstrates that APEMO consistently enhances trajectory- In this paper, we introduce APEMO (Affect-aware Peak...
- [introduction, p. 1] Alignment in these settings is no our approach aligns interaction dynamics with evaluation-sensitive longer solely a property of model parameters; it unfolds across time. trajectory properties.
- [introduction, p. 1] Over extended trajectories, ac- Figure 1 illustrates the core intuition of our approach.

**Result evidence**

- [results, p. 5] APEMO improves endpoint delivery quality by +0.1243 (95% CI [0.0522, 0.1924], 𝑝 = 0.0118) relative to task_peak_end.
- [results, p. 5] The CI and sign-test discrepancy on Relative to the peak-end baseline, APEMO improves mean tra- rebound is expected under this conservative test with mixed direc- jectory quality by +0.0791 (95% CI [0.0525, 0.1055], sign-test 𝑝 = tional wins.
- [results, p. 5] Relative to plain multi-agent flow, APEMO improves quality by +0.1782 (95% CI [0.1386, 0.2132], 𝑝 = 3.05 × 10−5 ) and reuse Hanjing Shi and Dominic DiFranzo Figure 3: Forest-style effect plot for key deltas (APEMO minus baseline) with 95% bootstrap CIs.

**Limitations / caution evidence**

- [limitations, p. 8] Several boundaries remain. of model training or interface transparency.
- [limitations, p. 8] It is also shaped by Trap-recovery micro-dynamics remain variance-sensitive in cer- how computation is distributed across time.
- [limitations, p. 8] While endpoint stabilization is robust, saliency as a controllable signal opens a complementary pathway rebound-style metrics fluctuate with model family and perturbation for engineering resilient agentic systems. profile.

### [2603.10044: Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety](../reading-cards/2603.10044.md)

- Class: `direct-competitor`; pages: 74; local text characters: 244056
- Full text: [PDF](../papers-all/2603.10044.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=6; MRT-Harness=0; Harness-C=16; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [conclusion, p. 62] The scaffold×benchmark interaction term (η 2 = 1.2%) is roughly three times the scaffold main effect itself, which confirms that scaffold impact is benchmark- specific rather than generic — map-reduce degrades TruthfulQA by approximately 20 pp on average yet improves XSTest by approximately 5 pp (Ta...
- [abstract, p. 1] Two of the three (ReAct and multi- agent) sit inside the pre-registered ±2 pp equivalence margin in pooled estimates — ReAct’s effect is statistically significant on Holm correction but still TOST-equivalent, and multi-agent is non-significant and TOST-equivalent; map-reduce delegation degrades pool...
- [abstract, p. 1] Properties with high baseline safety rates survive scaffolding; syco- phancy, the property with by far the lowest baseline (29.2% non-sycophantic at direct API), is also the only one where all three scaffolds improve safety on aggregate (+2.1 to +2.5 pp), and where model-by-scaffold heterogeneity re...

**Limitations / caution evidence**

- [limitations, p. 4] 7.5 Builder-as-Subject Validity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
- [limitations, p. 4] Consequential Safety Properties . . . . . . . . . . . . . . . . . . . . . .
- [limitations, p. 4] 41 7.7 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

### [2606.01416: Self-Healing Agentic Orchestrators for Reliable Tool-Augmented Large Language Model Systems](../reading-cards/2606.01416.md)

- Class: `direct-competitor`; pages: 33; local text characters: 87756
- Full text: [PDF](../papers-all/2606.01416.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=12; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] We propose a modular self-healing agentic orchestrator that implements a monitor–detect– diagnose–recover–verify loop.

**Result evidence**

- [results, p. 15] Self-healing achieves the highest overall task success rate, reaching 98.8%, compared with 94.5% for retry-only, 94.1% for the ReAct-style baseline, 93.8% for full replanning, and 70.1% for static workflow.
- [results, p. 15] This is a 4.3 percentage-point improvement over the strongest non-self-healing baseline, retry-only, and a 5.0 percentage-point improvement over full replanning.
- [experiments, p. 11] We evaluate whether self-healing orchestration improves reliability under tool, context, recovery, and semantic failure conditions.

**Limitations / caution evidence**

- [limitations, p. 25] The controlled benchmark is designed to isolate orchestration behavior under reproducible task, tool, and fault conditions.
- [limitations, p. 25] The compact model-in- 25 the-loop experiments add live model participation, but they still use local deterministic tools and controlled faults.
- [limitations, p. 25] The results therefore support claims about failure-aware orchestration under the evaluated conditions; they do not by themselves establish production reliability for arbitrary LLM agents, external APIs, user populations, or deployment environments.

## sequential-causal

### [2107.03544: The Micro-Randomized Trial for Developing Digital Interventions: Experimental Design and Data Analysis Considerations](../reading-cards/2107.03544.md)

- Class: `core-selected`; pages: 87; local text characters: 153221
- Full text: [PDF](../papers-all/2107.03544.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=33; Harness-C=4; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [abstract, p. 4] In this paper we review the micro-randomized trial (MRT), a study design that can be used to improve mobile health interventions by answering the above questions.
- [abstract, p. 4] Thus, it is good practice to limit content delivered by push intervention components to the minimum needed to achieve the desired effect.

**Limitations / caution evidence**

- [limitations, p. 56] MRT FOR DEVELOPING DIGITAL INTERVENTIONS 57 More work is needed to integrate MRTs into the general MOST framework.
- [limitations, p. 56] In this article we discussed the role of the MRT in optimization of decision rules for individual components of a digital intervention.
- [limitations, p. 56] However, in MOST the ultimate goal is optimization of the intervention as a whole, rather than optimization of individual intervention components (although the latter may be a useful step along the way).

### [1711.03587: The stratified micro-randomized trial design: sample size considerations for testing nested causal effects of time-varying treatments](../reading-cards/1711.03587.md)

- Class: `core-selected`; pages: 65; local text characters: 146550
- Full text: [PDF](../papers-all/1711.03587.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=30; Harness-C=8; ActiveHarness=0

**Contribution evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Result evidence**

- [results, p. 1] The calculator requires the scientist to specify a generative model for the his- tory Ht which achieves the specified alternative treatment effect.
- [experiments, p. 26] For each (W, Z) ∈ B(,0 ) , we wish to compute the achieved power under the alternative generative model where Vt under no treatment evolves as a Markov chain with transition matrix P constructed from inputs W and Z.
- [experiments, p. 26] Table 3 presents achieved power under the previously calculated sample sizes for Ω(0.02,4) and Ω(0.01,2) respectively.

**Limitations / caution evidence**

- [related_work, p. 3] Recently micro-randomized trial designs [Liao et al., 2016, Dempsey et al., 2015] were developed for testing proxi- mal and delayed effects of treatment [Klasnja et al., 2015].
- [related_work, p. 3] While in these trials treatment is sequentially randomized per participant, this approach does not permit the randomization probabilities to depend on features of the participant’s observation history.
- [related_work, p. 3] Indeed due to the rapid increase in sensor technology and the ability of various machine learning methods to provide real-time predictions, it is now feasible for scientists to trigger treatments based on these predictions or other features of the participant’s observation history.

### [2605.17641: Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents](../reading-cards/2605.17641.md)

- Class: `core-selected`; pages: 12; local text characters: 56251
- Full text: [PDF](../papers-all/2605.17641.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=20; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We propose Causal Memory memory mechanisms are central to personalized assis- Intervention (CMI), a causal memory-selection tants, autonomous research agents, interactive tutoring sys- technique that estimates how candidate memories tems, and long-horizon task-completion workflows, and are affect th...
- [introduction, p. 1] These bank is not a passive transcript but an actively retrieved and results suggest that effective long-term memory requires curated source of evidence. more than retrieving or compressing context: agents must select memories according to their causal effect on the cur- In this paper, we propose Ca...
- [introduction, p. 1] Rather than selecting memories only This work makes the following contributions: by semantic similarity (Lewis et al., 2020; Karpukhin et al., 2020) or compressed summaries, CMI estimates whether • We introduce Causal Memory Intervention (CMI), a a candidate memory causally improves the agent’s answ...

**Result evidence**

- [experiments, p. 6] This baseline tests whether simple graph gies under the same task, memory bank, response model, structure over memory text and memory scope improves and scoring pipeline.
- [experiments, p. 6] It first proposes candidate mem- implementation treats the full memory bank as selected ories and then evaluates them through no-memory, with- for metric computation, even though full-history prompts memory, and perturbed-memory intervention conditions. with past sessions and summary memory prompts...
- [conclusion, p. 11] Rather than retrieving memo- Learning Research, 26(72):1–75, 2025. ries solely because they are semantically similar to the cur- rent request, CMI estimates whether candidate memories Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.- improve the model’s answer under controlled intervention W.

**Limitations / caution evidence**

- [limitations, p. 10] filtering may suppress sensitive but relevant context, while This work studies CMI in a controlled, annotated memory- overly permissive filtering may reinforce stale, incorrect, or selection setting, and the results should be interpreted ac- biased memories.
- [limitations, p. 10] In the current implementation, memory entries useful for robustness evaluation, but similar constructions may include role annotations such as useful, irrelevant, or could be misused to attack memory-augmented agents (Liu harmful, and some methods use these annotations during se- et al., 2023; Zou e...
- [limitations, p. 10] Thus, our experiments evaluate the value of explicit therefore view CMI as one component of a broader re- causal-memory structure rather than proving that agents sponsible memory architecture, which should include data can infer causal roles from raw memories alone.

### [2002.06673: Performative Prediction](../reading-cards/2002.06673.md)

- Class: `core-selected`; pages: 32; local text characters: 82566
- Full text: [PDF](../papers-all/2002.06673.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=8; ActiveHarness=1

**Contribution evidence**

- [abstract, p. 1] We develop a risk minimization framework for performative prediction bringing together concepts from statistics, game theory, and causality.
- [introduction, p. 1] This mapping from predictive model to distribution is the key conceptual device of our framework.
- [method, p. 4] In this section, we formally introduce the principal solution concepts of our framework: perfor- mative optimality and performative stability.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [related_work, p. 3] Performativity is a broad concept in the social sciences, philosophy, and economics [18, 30].
- [related_work, p. 3] Below we focus on the relationship of our work to the most relevant technical scholarship.
- [related_work, p. 3] A closely related line of work considers the prob- lem of concept drift, broadly defined as the problem of learning when the target distribution over instances drifts with time.

### [2606.08275: Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures](../reading-cards/2606.08275.md)

- Class: `core-selected`; pages: 5; local text characters: 13493
- Full text: [PDF](../papers-all/2606.08275.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] We present Causal Agent Replay (CAR), which answers the question by interven- tion: it models an agent run as a structural causal model, applies a do(·) operation to a step, and re-executes the trajectory forward under the same stochastic policy, measuring the shift in the outcome distribution.

**Result evidence**

- No cue sentence extracted; inspect the reading card and full text.

**Limitations / caution evidence**

- [limitations, p. 5] The contrastive effect is a total effect through a stochastic continuation; isolating a step’s direct effect calls for common random numbers across branches, which is hard across divergent LLM contexts and is left as a refinement.
- [limitations, p. 5] Judge-based outcome functions inject their own noise; rule- based outcomes are preferred for anything to be trusted.
- [limitations, p. 5] Real tools with side effects are out of scope (the demonstrations use mocked, reproducible tools).

### [2606.00765: FALAT: Tracing Failures in LLM Agent Trajectories via Dependency-Guided Search](../reading-cards/2606.00765.md)

- Class: `core-selected`; pages: 17; local text characters: 57946
- Full text: [PDF](../papers-all/2606.00765.pdf)
- Five-idea scores: PREQ-Harness=0; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [introduction, p. 1] Our contributions are summarized as follows: • We formulate failure attribution in LLM agent trajectories as hierarchical dependency- guided search, shifting the task from flat step classification to structured diagnosis over candidate steps and their dependencies. • We propose FALAT, a diagnostic f...
- [conclusion, p. 10] Based on this formulation, we proposed FALAT, a four-stage framework consisting of search space construction with a task- conditioned prior, candidate pruning through typed dependencies, dependency-guided search and verification, and local re-search.

**Result evidence**

- [conclusion, p. 10] We evaluated FALAT on the Who&When benchmark, where it consistently outperformed existing SOTA baselines across multiple LLM backbones.
- [abstract, p. 1] The results show that FALAT consis- tently improves responsible-agent and decisive-step attribution.
- [abstract, p. 1] Its best configurations achieve 46.0% step-level accuracy on algorithm-generated trajectories and 29.1% on the more challeng- ing hand-crafted trajectories, outperforming specialized attribution baselines and direct prompt- ing with standalone LLMs.

**Limitations / caution evidence**

- [limitations, p. 9] FALAT relies on LLM-based judgments at multiple stages, in- cluding candidate selection, transition typing, and counterfactual verification.
- [limitations, p. 9] As a result, its effec- tiveness depends on the underlying model’s reasoning reliability, and errors in intermediate judg- ments may propagate through the pipeline.
- [limitations, p. 9] FALAT constructs hierarchical abstractions and typed dependencies using window-based processing, which may miss long-range dependencies that span across windows.

## transfer-generalization

### [1506.02629: Generalization in Adaptive Data Analysis and Holdout Reuse](../reading-cards/1506.02629.md)

- Class: `core-selected`; pages: 29; local text characters: 97429
- Full text: [PDF](../papers-all/1506.02629.pdf)
- Five-idea scores: PREQ-Harness=28; Harness Transport=0; MRT-Harness=0; Harness-C=0; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] Finally, we demonstrate that these incomparable approaches can be unified via the notion of approximate max-information that we introduce.
- [introduction, p. 1] 1.1 Our Results We propose a simple and general formulation of the problem of preserving statistical validity in adaptive data analysis.

**Result evidence**

- [results, p. 3] Using results from [DFH+ 14, NS15] we show that for datasets consisting of i.i.d. samples these modifications provably prevent the analyst from constructing functions that overfit to the holdout set.
- [experiments, p. 20] In this scenario no classifier can achieve true accuracy better than 50%.
- [experiments, p. 20] Indeed, a significant number of methods in the statistics and machine learning literature deal with inference for fixed two-step procedures where the first step is variable selection (see [HTF09] for examples).

**Limitations / caution evidence**

- [conclusion, p. 22] In this work, we give a unifying view of two techniques (differential privacy and description length bounds) which preserve the generalization guarantees of subsequent algorithms in adaptively chosen sequences of data analyses.
- [conclusion, p. 22] Although these two techniques both imply low max-information – and hence can be composed together while preserving their guarantees – the kinds of guarantees that can be achieved by either alone are incomparable.
- [conclusion, p. 22] This suggests that the problem of generalization guarantees under adaptivity is ripe for future study on two fronts.

### [2606.09498: Self-Harness: Harnesses That Improve Themselves](../reading-cards/2606.09498.md)

- Class: `core-selected`; pages: 19; local text characters: 58577
- Full text: [PDF](../papers-all/2606.09498.pdf)
- Five-idea scores: PREQ-Harness=10; Harness Transport=20; MRT-Harness=0; Harness-C=1; ActiveHarness=0

**Contribution evidence**

- [abstract, p. 1] In this paper, we introduce Self-Harness, a new paradigm in which an LLM-based agent improves its own operating harness, without relying on human engineers or stronger external agents.
- [conclusion, p. 13] We introduced Self-Harness, a propose–evaluate–accept framework in which the model is evaluated under the current harness, receives structured evidence from its own execution traces, and proposes bounded edits to declared harness surfaces.

**Result evidence**

- [results, p. 9] Across all three model backends, the promoted harness improves or preserves Pass (%) on both the held-in split and the held-out split.
- [results, p. 9] For MiniMax M2.5, Self- Harness improves held-in Pass from 43.0 to 50.0, a gain of 16% relative improvement, and improves held-out Pass from 40.5 to 61.9, a gain of 53% relative improvement.
- [results, p. 9] For Qwen3.5, Self-Harness improves held-in Pass from 15.1 to 36.0, a gain of 138% relative improvement, and held-out Pass from 23.8 to 38.1, a gain of 60% relative improvement.

**Limitations / caution evidence**

- [conclusion, p. 13] This paper studied whether a fixed language model can improve the harness that governs its own agent behavior.
- [conclusion, p. 13] We introduced Self-Harness, a propose–evaluate–accept framework in which the model is evaluated under the current harness, receives structured evidence from its own execution traces, and proposes bounded edits to declared harness surfaces.
- [conclusion, p. 13] Candidate harnesses are then re-evaluated under the same benchmark protocol, and only edits that satisfy a non-regressive acceptance rule are promoted into the harness lineage.

