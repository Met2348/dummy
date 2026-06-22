# L01 · 模型在商品化，harness 才是护城河

> Part I · 40-min lecture · 目标: 讲清「harness engineering」为什么在 2026 成为一门被正名的工程学科, 以及它为什么是你 PhD 值得关注的方向。

---

## 0. 一句话论点

> **2026 年，前沿模型已是商品；决定一个 agent 能不能在生产里干活的，是你给模型套的那层 harness。**

GPT-5.5、Claude Sonnet 4.6、Gemini 3.1 Pro、DeepSeek-V4 在几周内相继发布，能力都极强、且彼此接近。当「选哪个模型」不再是胜负手，竞争就转移到**模型外面那层基础设施**——loop、工具路由、上下文管理、记忆、护栏、可观测。这层就是 **harness**。

---

## 1. 什么是 harness（先把词说准）

> **harness（运行时外骨骼）**: 把一个 raw LLM 变成能干活 agent 的那层软件脚手架——反复调用模型、解析输出、执行工具、把结果喂回去、决定何时停，并在全程做上下文管理、权限校验、预算控制和追踪。

这个词借自软件测试里的 **test harness**（让你在受控条件下反复执行代码的脚手架）。2026 年它有了一个微妙的语义升级：

```
"scaffolding"(脚手架)              "harness"(外骨骼)
模型够强了就拆掉的临时结构    →    模型一直需要带在身上的永久结构
(早期叫法, 隐含 temporary)         (2026 叫法, 隐含 permanent)
```

这个转变本身就说明了行业认知的变化：**harness 不是模型成熟前的临时拐杖，而是模型永远需要的「神经系统 + 外骨骼」。** 我们花了几年把「大脑」(模型) 造得极强，却长期忽视让大脑能和真实世界交互的那套系统。

> 模型 vs harness 的分工 (本专题反复用的心智模型):
> - **模型** = 无状态的 token 预测器，提供「原始认知/推理」。
> - **harness** = 包裹模型的确定性运行时，**校验、授权、执行、记录**模型提出的每一个动作。
> - 一句话：**Model proposes, harness disposes.**（模型提议动作，harness 来校验 schema、权限、预算、安全后执行。）

---

## 2. 为什么它突然火了（2026 的两个数字）

harness engineering 不是炒概念，它是被**生产鸿沟**逼出来的。两个被反复引用的数字：

```
┌─────────────────────────────────────────────────────────┐
│  88%  的企业 agent 项目  ——  到不了生产                    │
│  65%  的失败             ——  根因是 harness 缺陷, 不是模型  │
│         (context drift / schema 错配 / state 退化)        │
└─────────────────────────────────────────────────────────┘
```

诊断很清楚：**失败大多不是「模型不够聪明」，而是「外面那层工程不行」。** 上下文漂移、工具 schema 对不齐、跨步骤 state 退化——这些都是 harness 的病，不是模型的病。**只优化模型而不稳住 harness，回报递减。**

这也解释了为什么 2026 年「能稳定上线 agent」的团队，都在**像系统工程师那样思考，而不是像 prompt 工程师**。

---

## 3. 这个词是怎么传开的（一段小历史）

- **2026-02**，HashiCorp 联合创始人 **Mitchell Hashimoto** 在一篇讲自己 AI 工作流的博客里用了 "harness engineering"。他的观点极简却深刻：**当 agent 犯了个错，不要去改 prompt 求它别犯——去改环境，让这个错根本不可能再发生。**
- 同一周，**OpenAI** 在 Codex 的工作里沿用了这个术语；**LangChain** 随即跟进同样的框架。
- 一个词在一周内被三方采用，标志着行业把「harness」从隐性的胶水代码，正式提升为**一等工程对象**。

> Hashimoto 那句话是整门学科的精神内核：**harness engineering 的本质，是把「改 prompt」的运气活，变成「改环境」的工程活。**

---

## 4. 你已经会的，和这门专题要加的

你在 Module 7 的 [agent-harness-design](../../agent-harness-design/) 里已经**从零搭过一个 mini-harness**，懂了 harness 由哪些零件组成 (loop / tools / context / permissions / trace)。那是**理解层**。

本专题是**工程层 + 前沿层**——把玩具 harness 推到能上生产的程度，并看见其中的研究 gap：

```
理解层 (你已会)                    工程层 + 前沿 (本专题)
─────────────────                 ──────────────────────────────
MockModel 跑通多步        →        接真模型 (provider 抽象) [L03]
基础 compaction          →        5 阶段渐进式 compaction [L04]
单会话                   →        跨窗口 long-horizon 自治 [L05]
trace + cost             →        OpenTelemetry 式可观测 [L09]
"harness 概念"           →        真 harness eval (同模型差几十分) [L10]
                                  + 研究入口: harness 的开放问题 [L13-14]
```

---

## 5. 为什么这对你（博0, NLP/LLM）尤其值得

harness engineering 是 2026 **少数「系统工程 ⨯ NLP 研究」真交叉**的方向：

- 它有真实的**开放研究问题**: long-horizon 自治、harness 评测方法、可移植 harness、context folding 的理论 (L13 会用你 `critical-reading-gap` 的 gap 雷达逐个扫)。
- 它有**学术化的苗头**: 70 个开源 agent 系统的实证研究、Claude Code 被逆向出的 5 阶段 compaction、把 harness 本身当研究对象的 *Natural-Language Agent Harnesses* (arXiv 2603.25723)。
- 它**门槛友好**: 不需要预训练大模型的算力，一台机器 + 一个 API（甚至像本专题这样用 Mock）就能做出真东西、测出真差异。

> 对一个要找研究方向的博0：这是一个**既火、又缺人系统做、又不烧算力**的方向。L13–L14 会带你把它从「热点」变成「你的候选题目」。

---

## 6. 本讲小结 + 通往 L02

- 模型商品化 → harness 是护城河。
- 88% 落地失败 / 65% 源于 harness 缺陷——它是被生产鸿沟逼出来的学科。
- 精神内核 (Hashimoto): 改环境，而不是改 prompt。
- 对你: 系统工程 ⨯ NLP 研究的稀缺交叉，门槛友好。

> **下一讲 L02**: 「harness」这个词现在被滥用——什么都叫 harness。我们需要一个**本构定义**: 到底哪 4 个要素，必要且充分地构成一个 harness？用它能严格鉴定 Claude Code / Codex / Aider / OpenHands 是不是真 harness，把「工具包装」「护栏」和「真 harness」区分开。

---

### 来源
- [The Agent Harness — NJ Raman (Medium, 2026-04)](https://medium.com/@nraman.n6/the-agent-harness-why-the-infrastructure-around-your-llm-is-more-important-than-the-llm-itself-3a6e5cbb2e97)
- [Harness Engineering: The Infrastructure Layer — Vishal Mysore (Medium, 2026-05)](https://medium.com/@visrow/harness-engineering-the-infrastructure-layer-that-makes-ai-agents-actually-work-598a279c1c5f)
- [Agent Harness Engineering — O'Reilly Radar](https://www.oreilly.com/radar/agent-harness-engineering/)
- [Agent Harness Engineering — Adnan Masood (Medium, 2026-04)](https://medium.com/@adnanmasood/agent-harness-engineering-the-rise-of-the-ai-control-plane-938ead884b1d)
- [awesome-harness-engineering (GitHub)](https://github.com/ai-boost/awesome-harness-engineering)
