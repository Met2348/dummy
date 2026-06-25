# M9 · 自动科研（Auto-Research / AI Scientist）教学系列规划

> 老师的判断：**「2026，Coding 已被 Agent 接管；下一个是 research。」** 这份规划把「AI 自己做科研」拆成
> **8 个专题（9.1–9.8）**，仿本仓库 M1–M8 的房子风格：每专题 = 学习目标 + lectures + 绑定文献（[`papers/`](papers/INDEX.md) 的 40 篇）+ **3080Ti 可跑的 hands-on** + 桥接已学模块 + 毕业 capstone。
>
> **定位**：M9 是整套 LLM 全栈自学的**收官系列**——前 8 组（造/改/用模型、推理、评测、Agent、Infra）学到的一切，在这里组装成「一个会做研究的 Agent」。
>
> 设计沿用本仓库三原则：**① 手写可跑的缩小版**（不靠跑不动的重型栈）·**② 防踩坑**（每个 demo 配批判视角）·**③ V0/V1/V2 可验证**（runbook + 测试）。
> 状态：**9.5 已落地为可跑模块** → [`m9.5-end-to-end-ai-scientist/`](m9.5-end-to-end-ai-scientist/)（V0/V1/V2 全绿，3080Ti 真训练）；其余 7 专题待续 · 2026-06-25

---

## 总览

| 专题 | 主题 | 核心问题 | 绑定文献（主题组） | 桥接已学模块 | 预估 |
|------|------|---------|------------------|------------|------|
| **9.1** | 自主性阶梯与全景 | research 自动化到哪一步了？ | A 综述 + G 批判 | agent-foundations | ~6h |
| **9.2** | 研究 Agent 的内核 | 一个研究 agent 由什么构成？ | B（co-scientist/v2/Agent Lab 的架构） | agent-foundations · tool-use-mcp · multi-agent-orchestration · agent-memory-context | ~8h |
| **9.3** | 创意与假设生成 | 科研第一步：点子从哪来、好不好？ | C 创意 | rag-essential · **llm-judge-arena** | ~7h |
| **9.4** | Deep Research / 文献综述 | 怎么自动读完一个领域并写出带引用的综述？ | D 文献合成 | rag-essential | ~7h |
| **9.5** ✅ | 端到端 AI Scientist ⭐（**已建可跑**） | 把 idea→实验→论文 整条链跑通 | B 端到端 | pretraining-recipe · rl-foundations · small-model-graduation | ~12h（capstone） |
| **9.6** | 评测 Research Agent | 怎么知道它"真做出了研究"而非刷分？ | E benchmark | **agent-code-eval** · eval-foundations · reasoning-eval | ~8h |
| **9.7** | 自我改进 / 自动算法发现 | 能不能让 AI 改进 AI 自己？ | F 自我改进 | rl-sota-2026 · process-reward | ~9h |
| **9.8** | 批判·安全·科研诚信 ⭐⭐ | 它会怎么骗你？怎么防？ | G 批判 | red-team-jailbreak · safety-defense · eval-graduation | ~10h（毕业 capstone） |

> 总时长 ~67h（约 8–10 周，每周 7–8h）。**先修**：建议先过 agent-foundations / rag-essential / agent-code-eval（你正好在学）。

---

## 9.1 自主性阶梯与全景

- **学习目标**：建立坐标系——能把任意一个"AI 科研系统"准确归到 **Tool / Analyst / Scientist** 哪一级；说清科研生命周期（创意→假设→设计→实验→分析→写作→评审）每一环的自动化现状；建立"乐观×批判"双视角。
- **Lectures（~8 讲）**：L01 为什么 research 是下一个前沿 · L02 Tool→Analyst→Scientist 三级阶梯（精读综述 2505.13259）· L03 科研生命周期七环 · L04 全景地图：30+ 系统鸟瞰 · L05 影响力 vs 时效：怎么读这个爆炸的领域 · L06 **现实检查**：hype 与真实能力的差距 · L07–08 综述精读工坊（A 组 6 篇）。
- **绑定文献**：A 组全部（2505.13259 主读）+ G 组先扫一遍建立警惕。
- **Hands-on（3080Ti/CPU）**：写一个 `taxonomy_classifier.py`——给它一段系统描述，用规则+LLM 把它归到三级阶梯并打分；对 INDEX 里的 40 篇做一次自动归类，产出一张领域地图。
- **桥接**：agent-foundations（你注释的 ReAct 就是 Tool→Analyst 的最小内核）。
- **产物**：一张"自主性阶梯 × 生命周期"二维地图（把 40 篇钉上去）。

## 9.2 研究 Agent 的内核

- **学习目标**：拆解一个研究 agent 的骨架——**规划（含树搜索）+ 工具使用（检索/代码执行/arXiv API）+ 记忆 + 多智能体角色**；理解 AI Scientist-v2 的"实验经理 agent + 树搜索"、co-scientist 的"Generation/Reflection/Evolution/Proximity 四agent"。
- **Lectures（~10 讲）**：L01 ReAct 回顾与放大 · L02 规划：Plan-and-Solve · L03 **Agentic 树搜索**（AI Scientist-v2）· L04 工具层：检索/代码执行/学术图谱 · L05 记忆与长上下文 · L06 多智能体角色分工 · L07 生成-辩论-进化（co-scientist）· L08 实验经理 agent · L09–10 实操工坊。
- **绑定文献**：2504.08066（树搜索+实验经理）· 2502.18864（多 agent 架构）· 2501.04227（PhD/Postdoc/Professor 角色）。
- **Hands-on**：把你已注释的 `react_loop` 扩成 `mini_research_agent.py`——问题→（mock 或真）arXiv 检索→拟 idea→自我批判→产出结构化研究计划。纯 CPU 可跑，留 LLM 接口可插真模型。
- **桥接**：agent-foundations · tool-use-mcp · multi-agent-orchestration · agent-memory-context（四个 M7 模块在这里合体）。

## 9.3 创意与假设生成

- **学习目标**：理解"创意"为何是 litmus test；**新颖性 vs 可行性**的张力；idea 排序/锦标赛；以及最关键的清醒认知——**ideation-execution gap（看着新 ≠ 做得出）**。
- **Lectures（~8 讲）**：L01 ideation 为何是第一步 · L02 精读 Si et al. 人类对照实验（2409.04109）· L03 **ideation-execution gap**（2506.20803，必讲）· L04 ResearchAgent 的 ReviewingAgents · L05 组合式创造 · L06 概念网络上的深度构想 · L07 idea 锦标赛与 Elo 排序 · L08 工坊。
- **绑定文献**：C 组全部 + co-scientist 的 tournament evolution。
- **Hands-on**：`idea_tournament.py`——生成 K 个 idea → RAG 检索相关工作 → 用 novelty 启发式 + LLM judge 打分排序 → 锦标赛选出 top-k。**直接复用 llm-judge-arena 的评委**。
- **桥接**：rag-essential（检索增强 idea）· **llm-judge-arena**（idea 排序就是 LLM-as-judge）。
- **防踩坑**：自评不可靠、多样性差——demo 里要显式展示"LLM 给自己点子打高分"的偏差。

## 9.4 Deep Research / 文献综述 Agent

- **学习目标**：掌握"自动读完一个领域并写出带引用综述"的范式：STORM 的多视角提问 + outline-driven RAG；OpenScholar 的检索库；**引用存在 ≠ 引用忠实**这一致命区分。
- **Lectures（~8 讲）**：L01 deep research 范式 · L02 精读 STORM（2402.14207）· L03 Co-STORM 思维导图 · L04 OpenScholar：4500 万论文库 + 引用准确率 · L05 **引用忠实性**与幻觉引用 · L06 评测：DeepScholar-Bench · L07 跑通开源 STORM/GPT-Researcher · L08 工坊。
- **绑定文献**：D 组全部。
- **Hands-on**：① 本机跑开源 STORM（`pip install knowledge-storm`）就一个真实主题产出综述；② 写 `mini_storm.py`——多视角问题生成→检索→带 inline 引用合成→**自动核查引用忠实度**（每句话回查源是否真支持）。
- **桥接**：rag-essential · agent-code-eval（"核查"的评测思维）。
- **现实意义**：这是 8 个专题里**当天就能用在你自己科研上**的一个。

## 9.5 端到端 AI Scientist ⭐（系列脊梁）

> ✅ **已落地为可跑模块**：[`m9.5-end-to-end-ai-scientist/`](m9.5-end-to-end-ai-scientist/) —— 五阶段真训练闭环、诚实光谱
> （supported/refuted/inconclusive）、grading-own-homework 现场实证、7 个诚实性测试 + runbook，V0/V1/V2 全绿。
> 一键跑：`python scripts/eric_3080ti_env_audit.py --runbook --modules auto-research-frontier/m9.5-end-to-end-ai-scientist`

- **学习目标**：把前面所有零件组装成 **idea→写码→实验→画图→写论文→自评审** 的完整闭环；理解 v1 模板 → v2 树搜索的演进、AI-Researcher 的"数学↔代码双向映射降幻觉"、执行 LLM 代码的**安全沙箱**。
- **Lectures（~12 讲）**：L01 五阶段总览 · L02 v1 模板法 · L03 v2 树搜索+VLM 看图 · L04 AI-Researcher 双向映射 · L05 NovelSeek 闭环 · L06 写码与实验执行 · L07 自动写作 · L08 自动评审（及其循环性）· L09 成本与沙箱安全 · L10 AgentRxiv 协作 · L11–12 **Capstone 搭建**。
- **绑定文献**：B 组全部。
- **Hands-on / Capstone**：`mini_ai_scientist/`——在一个 toy ML 问题上跑通端到端：自动拟 idea → 生成一个小训练脚本 → **在 3080Ti 上真跑** → 画图 → 自动写 1 页报告。**直接复用** pretraining-recipe / rl-foundations / small-model-graduation 的训练件。
- **桥接**：几乎整个仓库都在这里汇流（造模型→改模型→用模型→评测）。
- **防踩坑**：实验数太少→结论不可靠；会幻觉整张消融表（接 9.8 红队）。

## 9.6 评测 Research Agent

- **学习目标**：把"评测"建成方法论护城河——区分 **复现类**（PaperBench/CORE-Bench）·**工程类**（MLE-bench/RE-Bench）·**开放研究类**（MLR-Bench/ScienceAgentBench/MLGym）；读懂 RE-Bench 的"时间预算决定人机胜负"；会**自己设计**一个 research-agent benchmark。
- **Lectures（~9 讲）**：L01 为何 eval 是护城河 · L02 复现 vs 工程 vs 开放 三类 · L03 精读 RE-Bench（人机对照）· L04 PaperBench 复现评分 · L05 MLE-bench/Kaggle · L06 MLGym 的 Gym 环境 · L07 数据驱动（ScienceAgentBench）· L08 可复现性（CORE-Bench）· L09 设计你自己的基准。
- **绑定文献**：E 组全部。
- **Hands-on**：`mini_replication_eval.py`——给 agent 一份"方法 spec + 评分 rubric"，自动判分一个候选实现。**直接扩展 agent-code-eval 的 `safe_exec`**（你刚验证过的真沙箱）。
- **桥接**：**agent-code-eval（最直接——SWE-bench/HumanEval 的 safe_exec 就是这里的地基）** · eval-foundations · reasoning-eval。

## 9.7 自我改进 / 自动算法发现

- **学习目标**：理解最激进的一支——**AlphaEvolve（进化"解"）vs Darwin Gödel Machine（进化"自己"）vs ADAS（agent 设计 agent）**；理解 fitness/reward 信号及其**被刷**的方式；档案库与开放式进化搜索。
- **Lectures（~9 讲）**：L01 理论 Gödel Machine → 经验版 · L02 精读 DGM（2505.22954）· L03 AlphaEvolve · L04 ADAS：搜索 agent 架构 · L05 fitness 信号设计 · L06 **fitness 被 game**（接 process-reward 你已知的 reward hacking）· L07 档案库与多样性 · L08–09 工坊。
- **绑定文献**：F 组全部。
- **Hands-on**：`mini_self_improve.py`——一个 base agent 改写自己的 prompt/小段代码 → 在小 benchmark 上评估 → keep-if-better → 存档案；观察它怎么"作弊式"涨分。CPU/3080Ti 可跑。
- **桥接**：rl-sota-2026 · process-reward（fitness = reward，你已知它多易被刷）· rl-foundations。

## 9.8 批判·安全·科研诚信 ⭐⭐（毕业 capstone）

- **学习目标**：建立**判断力护城河**——看穿"自己给自己打分""幻觉结果/消融""刷简单基准""换小数据集"；掌握可复现性守卫（artifact 提交、执行日志、独立验证）；理解执行 LLM 代码的安全与治理（评审洪水、诚信）。
- **Lectures（~9 讲）**：L01 grading-its-own-homework · L02 幻觉结果与消融 · L03 benchmark gaming/数据替换（精读 Hidden Pitfalls 2509.08713）· L04 ideation-execution gap 回看 · L05 "Why LLMs Aren't Scientists Yet"（2601.03315）· L06 可复现性守卫 · L07 执行 LLM 代码的沙箱安全 · L08 治理与伦理 · L09 毕业答辩。
- **绑定文献**：G 组全部 + 各系统 Limitations 节。
- **Hands-on / 毕业 Capstone**：拿 9.5 的 `mini_ai_scientist`，**红队它**——找出它会在哪刷分/幻觉；加守卫（断言真实数据集、记录执行日志、加独立 verifier 复算）；写一份批判报告。**毕业标准**：你的 AI-Scientist 既能跑通，又扛得住自己的红队。
- **桥接**：red-team-jailbreak · safety-defense · eval-graduation（M6 尾三件套在这里收口）。

---

## 系列毕业 Capstone（贯穿 9.5+9.6+9.8）

> **在 3080Ti 上复现一个"可信的 mini AI-Scientist"**：端到端跑通（9.5）+ 自带复现评测（9.6）+ 扛得住红队的诚信守卫（9.8）。
> 交付：可跑代码 + runbook.yaml（V0/V1/V2）+ 一份"它能做什么/会怎么骗你"的诚实报告。这就是你对"research 被 Agent 接管到什么程度"的亲手答案。

## 设计原则（沿用本仓库）

1. **手写可跑的缩小版**：所有 mini_* 都是单进程、纯 CPU 或单卡可跑的真实现，不依赖跑不动的重型栈——和你验证过的 26+ 模块同款哲学。
2. **每个 demo 配批判**：乐观系统与对应批判文献成对教（如 9.5 端到端 ↔ 9.8 红队），训练科研品味。
3. **V0/V1/V2 可验证**：每专题产出 `runbook.yaml`，进得了 `eric_3080ti_env_audit.py --runbook`，和主仓库一套验证标准。

## 与已学模块的依赖图

```
M1 PEFT ─ M2 数据 ─ M3 造模型 ─ M4 改模型 ─ M5 用模型 ─┐
                                                      ├─► 9.5 端到端 AI Scientist
M7 Agent（ReAct/RAG/tool/multi-agent/memory）─────────┘        │
M6 评测（agent-code-eval/eval/reasoning）──► 9.6 评测 ──────────┤
M4 RL/process-reward ──► 9.7 自我改进 ─────────────────────────┤
M6 安全（red-team/safety/eval-graduation）──► 9.8 批判与诚信 ───┘
```

> 一句话：**M9 不是新起点，是你整套自学的"毕业设计"。** 你缺的主要是 9.7（进化搜索）和"科学方法论"本身，其余都是已学件的重组。

---

## 📌 进度与下一步

- ✅ **9.5 端到端**：已建成可跑可验证模块（脊梁先立起来了）。
- **建议的下一专题**（择一，我继续按同样标准建）：
  - **9.6 评测**：和 9.5 最配——给 9.5 的产物做"复现/独立验证"，直接扩你已验证的 `agent-code-eval` safe_exec；也补上 9.5 里 `review` 那条"裁判=选手"的根本循环性。
  - **9.8 批判·红队**：拿 9.5 的 mini scientist 当靶子，红队它（刷分/幻觉/换数据），加复现守卫——把脊梁变"可信"。
  - **9.4 Deep Research**：换个方向，最快用在你自己的文献综述（当天跑 STORM）。
- 也可以**就地深化 9.5**：把 `analysis.verdict()` 升级成真 t-检验（README 留的练习），或把 `ideation`/`review` 接上真 LLM。

> 告诉我下一个专题，我接着把"规划"变成"可跑模块"。
