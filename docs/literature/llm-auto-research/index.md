# LLM Auto Research / AI Scientist 文献包（70 篇）

本目录收集 70 篇 LLM 自动科研、AI Scientist、AI co-scientist、Deep Research、agent benchmark、工具调用、代码实验 agent、可验证发现和领域科学 agent 相关论文。每篇论文有一份独立的 30 分钟组会报告。

## 读法建议

1. 先读 A 组，建立端到端 AI Scientist 和真实科学发现图谱。
2. 再读 B 组，补齐老师要求的 deep literature research / auto survey / deep research agent。
3. 接着读 C 组，理解 idea generation、novelty illusion 和 LLM judge 风险。
4. 然后读 D-E 组，理解自动科研背后的 agent、tool、code、data science 和实验执行能力。
5. 最后读 F 组，理解通用 agent benchmark 与真实科研 benchmark 的差距。

## 文件结构

- `papers/`: 本地下载的 PDF 或 HTML 正文。
- `reports/`: 每篇论文一份 30 分钟组会报告。
- `metadata/papers_manifest.json`: 70 篇论文元数据和来源链接。
- `metadata/download_status.json`: 下载状态。
- `metadata/paper_frontmatter.json`: 从本地文件抽取的前文文本，便于后续二次加工。
- `metadata/new_31_70_arxiv_metadata.json`: 本轮新增 40 篇的 arXiv 元数据快照。

## 分组阅读路线

### A. 端到端 AI Scientist 与真实科学发现

- 01. [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](reports/01-ai-scientist-group-meeting-report.md)
- 02. [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](reports/02-ai-scientist-v2-group-meeting-report.md)
- 03. [Agent Laboratory: Using LLM Agents as Research Assistants](reports/03-agent-laboratory-group-meeting-report.md)
- 04. [AI-Researcher: Autonomous Scientific Innovation](reports/04-ai-researcher-group-meeting-report.md)
- 05. [DeepScientist: Advancing Frontier-Pushing Scientific Findings Progressively](reports/05-deep-scientist-group-meeting-report.md)
- 06. [AgentRxiv: Towards Collaborative Autonomous Research](reports/06-agentrxiv-group-meeting-report.md)
- 07. [Towards an AI co-scientist](reports/07-ai-co-scientist-group-meeting-report.md)
- 08. [Robin: A multi-agent system for automating scientific discovery](reports/08-robin-group-meeting-report.md)
- 09. [An AI system to help scientists write expert-level empirical software](reports/09-era-group-meeting-report.md)
- 25. [Autonomous chemical research with large language models](reports/25-coscientist-group-meeting-report.md)
- 27. [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](reports/27-sciagents-group-meeting-report.md)
- 28. [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](reports/28-virtual-lab-group-meeting-report.md)
- 29. [Mathematical discoveries from program search with large language models](reports/29-funsearch-group-meeting-report.md)
- 30. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](reports/30-alphaevolve-group-meeting-report.md)

### B. 文献调研、自动综述与 Deep Research

- 10. [ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models](reports/10-researchagent-group-meeting-report.md)
- 11. [Language agents achieve superhuman synthesis of scientific knowledge](reports/11-paperqa2-group-meeting-report.md)
- 12. [OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs](reports/12-openscholar-group-meeting-report.md)
- 31. [From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery](reports/31-from-automation-to-autonomy-group-meeting-report.md)
- 32. [AutoSurvey: Large Language Models Can Automatically Write Surveys](reports/32-autosurvey-group-meeting-report.md)
- 33. [SurveyX: Academic Survey Automation via Large Language Models](reports/33-surveyx-group-meeting-report.md)
- 34. [Agentic AutoSurvey: Let LLMs Survey LLMs](reports/34-agentic-autosurvey-group-meeting-report.md)
- 35. [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](reports/35-storm-group-meeting-report.md)
- 36. [Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations](reports/36-costorm-group-meeting-report.md)
- 37. [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](reports/37-deepresearch-bench-group-meeting-report.md)
- 38. [DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report](reports/38-deepresearch-bench-ii-group-meeting-report.md)
- 39. [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](reports/39-browsecomp-group-meeting-report.md)
- 40. [Characterizing Deep Research: A Benchmark and Formal Definition](reports/40-characterizing-deep-research-group-meeting-report.md)
- 41. [Deep Research: A Survey of Autonomous Research Agents](reports/41-deep-research-survey-group-meeting-report.md)
- 42. [BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent](reports/42-browsecomp-plus-group-meeting-report.md)
- 43. [MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents](reports/43-mmdeepresearch-bench-group-meeting-report.md)

### C. Idea generation、novelty 与风险

- 13. [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](reports/13-llm-novel-ideas-group-meeting-report.md)
- 14. [ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](reports/14-researchbench-group-meeting-report.md)
- 15. [IdeaBench: Benchmarking Large Language Models for Research Idea Generation](reports/15-ideabench-group-meeting-report.md)
- 16. [All That Glitters is Not Novel: Plagiarism in AI Generated Research](reports/16-all-that-glitters-group-meeting-report.md)
- 17. [On the Limits of LLM-as-Judge for Scientific Novelty Assessment](reports/17-llm-judge-novelty-limits-group-meeting-report.md)

### D. Agent 基础能力: reasoning、tool use、multi-agent、memory

- 44. [ReAct: Synergizing Reasoning and Acting in Language Models](reports/44-react-group-meeting-report.md)
- 45. [Reflexion: Language Agents with Verbal Reinforcement Learning](reports/45-reflexion-group-meeting-report.md)
- 46. [Toolformer: Language Models Can Teach Themselves to Use Tools](reports/46-toolformer-group-meeting-report.md)
- 47. [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](reports/47-autogen-group-meeting-report.md)
- 48. [CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](reports/48-camel-group-meeting-report.md)
- 49. [Generative Agents: Interactive Simulacra of Human Behavior](reports/49-generative-agents-group-meeting-report.md)
- 50. [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](reports/50-metagpt-group-meeting-report.md)
- 51. [Voyager: An Open-Ended Embodied Agent with Large Language Models](reports/51-voyager-group-meeting-report.md)
- 52. [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](reports/52-tree-of-thoughts-group-meeting-report.md)
- 53. [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](reports/53-graph-of-thoughts-group-meeting-report.md)
- 54. [Self-Refine: Iterative Refinement with Self-Feedback](reports/54-self-refine-group-meeting-report.md)
- 55. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](reports/55-self-consistency-group-meeting-report.md)
- 56. [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](reports/56-program-of-thoughts-group-meeting-report.md)
- 57. [PAL: Program-aided Language Models](reports/57-pal-group-meeting-report.md)
- 58. [Gorilla: Large Language Model Connected with Massive APIs](reports/58-gorilla-group-meeting-report.md)
- 59. [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](reports/59-hugginggpt-group-meeting-report.md)
- 60. [Executable Code Actions Elicit Better LLM Agents](reports/60-codeact-group-meeting-report.md)

### E. 代码、数据科学与实验执行 agent

- 20. [MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation](reports/20-mlagentbench-group-meeting-report.md)
- 21. [MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents](reports/21-mlr-copilot-group-meeting-report.md)
- 22. [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](reports/22-mle-bench-group-meeting-report.md)
- 23. [PaperBench: Evaluating AI's Ability to Replicate AI Research](reports/23-paperbench-group-meeting-report.md)
- 24. [Aviary: training language agents on challenging scientific tasks](reports/24-aviary-group-meeting-report.md)
- 61. [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](reports/61-swe-agent-group-meeting-report.md)
- 62. [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](reports/62-swe-bench-group-meeting-report.md)
- 68. [Data Interpreter: An LLM Agent For Data Science](reports/68-data-interpreter-group-meeting-report.md)
- 69. [DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation](reports/69-ds-1000-group-meeting-report.md)
- 70. [ChatDev: Communicative Agents for Software Development](reports/70-chatdev-group-meeting-report.md)

### F. 通用/网页/OS/助手 benchmark

- 18. [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](reports/18-scienceagentbench-group-meeting-report.md)
- 19. [DiscoveryBench: Towards Data-Driven Discovery with Large Language Models](reports/19-discoverybench-group-meeting-report.md)
- 63. [AgentBench: Evaluating LLMs as Agents](reports/63-agentbench-group-meeting-report.md)
- 64. [WebArena: A Realistic Web Environment for Building Autonomous Agents](reports/64-webarena-group-meeting-report.md)
- 65. [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](reports/65-osworld-group-meeting-report.md)
- 66. [GAIA: a benchmark for General AI Assistants](reports/66-gaia-group-meeting-report.md)
- 67. [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](reports/67-tau-bench-group-meeting-report.md)

## 70 篇论文索引

| # | 论文 | 年份 | 来源 | 类别 | 本地文件 |
|---:|---|---:|---|---|---|
| 01 | [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](reports/01-ai-scientist-group-meeting-report.md) | 2024 | arXiv | end_to_end | [本地文件](papers/01-ai-scientist.pdf) |
| 02 | [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](reports/02-ai-scientist-v2-group-meeting-report.md) | 2025 | arXiv | end_to_end | [本地文件](papers/02-ai-scientist-v2.pdf) |
| 03 | [Agent Laboratory: Using LLM Agents as Research Assistants](reports/03-agent-laboratory-group-meeting-report.md) | 2025 | arXiv | end_to_end | [本地文件](papers/03-agent-laboratory.pdf) |
| 04 | [AI-Researcher: Autonomous Scientific Innovation](reports/04-ai-researcher-group-meeting-report.md) | 2025 | arXiv | end_to_end | [本地文件](papers/04-ai-researcher.pdf) |
| 05 | [DeepScientist: Advancing Frontier-Pushing Scientific Findings Progressively](reports/05-deep-scientist-group-meeting-report.md) | 2025 | arXiv | end_to_end | [本地文件](papers/05-deep-scientist.pdf) |
| 06 | [AgentRxiv: Towards Collaborative Autonomous Research](reports/06-agentrxiv-group-meeting-report.md) | 2025 | arXiv | end_to_end | [本地文件](papers/06-agentrxiv.pdf) |
| 07 | [Towards an AI co-scientist](reports/07-ai-co-scientist-group-meeting-report.md) | 2025 | arXiv | hypothesis | [本地文件](papers/07-ai-co-scientist.pdf) |
| 08 | [Robin: A multi-agent system for automating scientific discovery](reports/08-robin-group-meeting-report.md) | 2025 | arXiv | domain_biomed | [本地文件](papers/08-robin.pdf) |
| 09 | [An AI system to help scientists write expert-level empirical software](reports/09-era-group-meeting-report.md) | 2025 | arXiv | experimentation | [本地文件](papers/09-era.pdf) |
| 10 | [ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models](reports/10-researchagent-group-meeting-report.md) | 2024 | arXiv | ideation | [本地文件](papers/10-researchagent.pdf) |
| 11 | [Language agents achieve superhuman synthesis of scientific knowledge](reports/11-paperqa2-group-meeting-report.md) | 2024 | arXiv | literature | [本地文件](papers/11-paperqa2.pdf) |
| 12 | [OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs](reports/12-openscholar-group-meeting-report.md) | 2024 | arXiv | literature | [本地文件](papers/12-openscholar.pdf) |
| 13 | [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](reports/13-llm-novel-ideas-group-meeting-report.md) | 2024 | arXiv | ideation_eval | [本地文件](papers/13-llm-novel-ideas.pdf) |
| 14 | [ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](reports/14-researchbench-group-meeting-report.md) | 2025 | arXiv | benchmark | [本地文件](papers/14-researchbench.pdf) |
| 15 | [IdeaBench: Benchmarking Large Language Models for Research Idea Generation](reports/15-ideabench-group-meeting-report.md) | 2024 | arXiv | benchmark | [本地文件](papers/15-ideabench.pdf) |
| 16 | [All That Glitters is Not Novel: Plagiarism in AI Generated Research](reports/16-all-that-glitters-group-meeting-report.md) | 2025 | ACL 2025 / arXiv | risk | [本地文件](papers/16-all-that-glitters.pdf) |
| 17 | [On the Limits of LLM-as-Judge for Scientific Novelty Assessment](reports/17-llm-judge-novelty-limits-group-meeting-report.md) | 2026 | arXiv | risk | [本地文件](papers/17-llm-judge-novelty-limits.pdf) |
| 18 | [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](reports/18-scienceagentbench-group-meeting-report.md) | 2024 | arXiv / ICLR 2025 | benchmark | [本地文件](papers/18-scienceagentbench.pdf) |
| 19 | [DiscoveryBench: Towards Data-Driven Discovery with Large Language Models](reports/19-discoverybench-group-meeting-report.md) | 2024 | arXiv / ICLR 2025 | benchmark | [本地文件](papers/19-discoverybench.pdf) |
| 20 | [MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation](reports/20-mlagentbench-group-meeting-report.md) | 2023 | arXiv / ICML 2024 | benchmark | [本地文件](papers/20-mlagentbench.pdf) |
| 21 | [MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents](reports/21-mlr-copilot-group-meeting-report.md) | 2024 | arXiv | experimentation | [本地文件](papers/21-mlr-copilot.pdf) |
| 22 | [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](reports/22-mle-bench-group-meeting-report.md) | 2024 | arXiv | benchmark | [本地文件](papers/22-mle-bench.pdf) |
| 23 | [PaperBench: Evaluating AI's Ability to Replicate AI Research](reports/23-paperbench-group-meeting-report.md) | 2025 | arXiv | benchmark | [本地文件](papers/23-paperbench.pdf) |
| 24 | [Aviary: training language agents on challenging scientific tasks](reports/24-aviary-group-meeting-report.md) | 2024 | arXiv | agent_training | [本地文件](papers/24-aviary.pdf) |
| 25 | [Autonomous chemical research with large language models](reports/25-coscientist-group-meeting-report.md) | 2023 | Nature | domain_chemistry | [本地文件](papers/25-coscientist.pdf) |
| 26 | [ChemCrow: Augmenting large-language models with chemistry tools](reports/26-chemcrow-group-meeting-report.md) | 2024 | Nature Machine Intelligence | domain_chemistry | [本地文件](papers/26-chemcrow.pdf) |
| 27 | [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](reports/27-sciagents-group-meeting-report.md) | 2024 | arXiv | domain_materials | [本地文件](papers/27-sciagents.pdf) |
| 28 | [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](reports/28-virtual-lab-group-meeting-report.md) | 2025 | Nature | domain_biomed | [本地文件](papers/28-virtual-lab.html) |
| 29 | [Mathematical discoveries from program search with large language models](reports/29-funsearch-group-meeting-report.md) | 2024 | Nature | verifiable_discovery | [本地文件](papers/29-funsearch.pdf) |
| 30 | [AlphaEvolve: A coding agent for scientific and algorithmic discovery](reports/30-alphaevolve-group-meeting-report.md) | 2025 | arXiv | verifiable_discovery | [本地文件](papers/30-alphaevolve.pdf) |
| 31 | [From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery](reports/31-from-automation-to-autonomy-group-meeting-report.md) | 2025 | arXiv | survey | [本地文件](papers/31-from-automation-to-autonomy.pdf) |
| 32 | [AutoSurvey: Large Language Models Can Automatically Write Surveys](reports/32-autosurvey-group-meeting-report.md) | 2024 | arXiv | survey_generation | [本地文件](papers/32-autosurvey.pdf) |
| 33 | [SurveyX: Academic Survey Automation via Large Language Models](reports/33-surveyx-group-meeting-report.md) | 2025 | arXiv | survey_generation | [本地文件](papers/33-surveyx.pdf) |
| 34 | [Agentic AutoSurvey: Let LLMs Survey LLMs](reports/34-agentic-autosurvey-group-meeting-report.md) | 2025 | arXiv | survey_generation | [本地文件](papers/34-agentic-autosurvey.pdf) |
| 35 | [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](reports/35-storm-group-meeting-report.md) | 2024 | arXiv | literature_synthesis | [本地文件](papers/35-storm.pdf) |
| 36 | [Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations](reports/36-costorm-group-meeting-report.md) | 2024 | arXiv | literature_synthesis | [本地文件](papers/36-costorm.pdf) |
| 37 | [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](reports/37-deepresearch-bench-group-meeting-report.md) | 2025 | arXiv | deep_research_benchmark | [本地文件](papers/37-deepresearch-bench.pdf) |
| 38 | [DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report](reports/38-deepresearch-bench-ii-group-meeting-report.md) | 2026 | arXiv | deep_research_benchmark | [本地文件](papers/38-deepresearch-bench-ii.pdf) |
| 39 | [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](reports/39-browsecomp-group-meeting-report.md) | 2025 | arXiv | browsing_benchmark | [本地文件](papers/39-browsecomp.pdf) |
| 40 | [Characterizing Deep Research: A Benchmark and Formal Definition](reports/40-characterizing-deep-research-group-meeting-report.md) | 2025 | arXiv | deep_research_definition | [本地文件](papers/40-characterizing-deep-research.pdf) |
| 41 | [Deep Research: A Survey of Autonomous Research Agents](reports/41-deep-research-survey-group-meeting-report.md) | 2025 | arXiv | deep_research_survey | [本地文件](papers/41-deep-research-survey.pdf) |
| 42 | [BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent](reports/42-browsecomp-plus-group-meeting-report.md) | 2025 | arXiv | browsing_benchmark | [本地文件](papers/42-browsecomp-plus.pdf) |
| 43 | [MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents](reports/43-mmdeepresearch-bench-group-meeting-report.md) | 2026 | arXiv | multimodal_deep_research_benchmark | [本地文件](papers/43-mmdeepresearch-bench.pdf) |
| 44 | [ReAct: Synergizing Reasoning and Acting in Language Models](reports/44-react-group-meeting-report.md) | 2022 | arXiv | agent_foundation | [本地文件](papers/44-react.pdf) |
| 45 | [Reflexion: Language Agents with Verbal Reinforcement Learning](reports/45-reflexion-group-meeting-report.md) | 2023 | arXiv | agent_learning | [本地文件](papers/45-reflexion.pdf) |
| 46 | [Toolformer: Language Models Can Teach Themselves to Use Tools](reports/46-toolformer-group-meeting-report.md) | 2023 | arXiv | tool_use | [本地文件](papers/46-toolformer.pdf) |
| 47 | [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](reports/47-autogen-group-meeting-report.md) | 2023 | arXiv | multi_agent_framework | [本地文件](papers/47-autogen.pdf) |
| 48 | [CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](reports/48-camel-group-meeting-report.md) | 2023 | arXiv | multi_agent_society | [本地文件](papers/48-camel.pdf) |
| 49 | [Generative Agents: Interactive Simulacra of Human Behavior](reports/49-generative-agents-group-meeting-report.md) | 2023 | arXiv | simulated_agents | [本地文件](papers/49-generative-agents.pdf) |
| 50 | [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](reports/50-metagpt-group-meeting-report.md) | 2023 | arXiv | multi_agent_framework | [本地文件](papers/50-metagpt.pdf) |
| 51 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](reports/51-voyager-group-meeting-report.md) | 2023 | arXiv | open_ended_agent | [本地文件](papers/51-voyager.pdf) |
| 52 | [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](reports/52-tree-of-thoughts-group-meeting-report.md) | 2023 | arXiv | reasoning_search | [本地文件](papers/52-tree-of-thoughts.pdf) |
| 53 | [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](reports/53-graph-of-thoughts-group-meeting-report.md) | 2023 | arXiv | reasoning_search | [本地文件](papers/53-graph-of-thoughts.pdf) |
| 54 | [Self-Refine: Iterative Refinement with Self-Feedback](reports/54-self-refine-group-meeting-report.md) | 2023 | arXiv | self_improvement | [本地文件](papers/54-self-refine.pdf) |
| 55 | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](reports/55-self-consistency-group-meeting-report.md) | 2022 | arXiv | reasoning | [本地文件](papers/55-self-consistency.pdf) |
| 56 | [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](reports/56-program-of-thoughts-group-meeting-report.md) | 2022 | arXiv | program_reasoning | [本地文件](papers/56-program-of-thoughts.pdf) |
| 57 | [PAL: Program-aided Language Models](reports/57-pal-group-meeting-report.md) | 2022 | arXiv | program_reasoning | [本地文件](papers/57-pal.pdf) |
| 58 | [Gorilla: Large Language Model Connected with Massive APIs](reports/58-gorilla-group-meeting-report.md) | 2023 | arXiv | tool_use_api | [本地文件](papers/58-gorilla.pdf) |
| 59 | [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](reports/59-hugginggpt-group-meeting-report.md) | 2023 | arXiv | tool_orchestration | [本地文件](papers/59-hugginggpt.pdf) |
| 60 | [Executable Code Actions Elicit Better LLM Agents](reports/60-codeact-group-meeting-report.md) | 2024 | arXiv | code_action_agent | [本地文件](papers/60-codeact.pdf) |
| 61 | [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](reports/61-swe-agent-group-meeting-report.md) | 2024 | arXiv | software_engineering_agent | [本地文件](papers/61-swe-agent.pdf) |
| 62 | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](reports/62-swe-bench-group-meeting-report.md) | 2023 | arXiv | software_engineering_benchmark | [本地文件](papers/62-swe-bench.pdf) |
| 63 | [AgentBench: Evaluating LLMs as Agents](reports/63-agentbench-group-meeting-report.md) | 2023 | arXiv | agent_benchmark | [本地文件](papers/63-agentbench.pdf) |
| 64 | [WebArena: A Realistic Web Environment for Building Autonomous Agents](reports/64-webarena-group-meeting-report.md) | 2023 | arXiv | web_agent_benchmark | [本地文件](papers/64-webarena.pdf) |
| 65 | [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](reports/65-osworld-group-meeting-report.md) | 2024 | arXiv | os_agent_benchmark | [本地文件](papers/65-osworld.pdf) |
| 66 | [GAIA: a benchmark for General AI Assistants](reports/66-gaia-group-meeting-report.md) | 2023 | arXiv | assistant_benchmark | [本地文件](papers/66-gaia.pdf) |
| 67 | [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](reports/67-tau-bench-group-meeting-report.md) | 2024 | arXiv | tool_agent_benchmark | [本地文件](papers/67-tau-bench.pdf) |
| 68 | [Data Interpreter: An LLM Agent For Data Science](reports/68-data-interpreter-group-meeting-report.md) | 2024 | arXiv | data_science_agent | [本地文件](papers/68-data-interpreter.pdf) |
| 69 | [DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation](reports/69-ds-1000-group-meeting-report.md) | 2022 | arXiv | data_science_benchmark | [本地文件](papers/69-ds-1000.pdf) |
| 70 | [ChatDev: Communicative Agents for Software Development](reports/70-chatdev-group-meeting-report.md) | 2023 | arXiv | software_dev_agent | [本地文件](papers/70-chatdev.pdf) |

## 方向总览

这 70 篇可以压缩成一个判断: LLM Auto Research 正在从 `literature assistant` 走向 `hypothesis + experiment + evaluation + writing + memory + collaboration` 的闭环系统。数量扩展后，图谱更清楚地分成三层: 第一层是文献和 Deep Research，第二层是 agent/tool/code 执行能力，第三层是 scientific discovery 的评价与验证。真正适合作为 PhD 切口的，不是泛泛做一个 AI Scientist，而是选定一个领域，例如 video generation / world models，然后建立可评测、可复现、可追溯的 AI research workflow。
