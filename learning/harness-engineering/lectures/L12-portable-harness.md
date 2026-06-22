# L12 · 可移植 harness 与 NL harness 前沿

> Part III · 40-min lecture · 目标: Part III 收尾 + 第一个明确的研究前沿——harness 逻辑为什么散落、不可移植, 以及把 harness 本身当一等研究对象的新工作。

---

## 0. 一个悖论

L11 刚证明: harness 能被**统一建模** (一个调度器 + 4 本构要素)。既然能统一建模, harness 理应是一个**清晰、可移植的工件** (artifact)——像一份配置、一个可复用的模块。

但现实恰恰相反:

> 尽管 harness 越来越重要, **harness 逻辑很少以一个连贯、可移植的工件形式存在**。在大多数 agent 系统里, 真正起作用的 harness **散落**在 controller 代码、隐藏的框架默认值、工具适配器、verifier 脚本、和运行时特有的假设里。

```
理想:  harness = 一个可移植工件          现实:  harness 散落各处
       ┌──────────────┐                      controller 代码里一点
       │  harness.yaml │                      框架默认值里一点 (你都不知道)
       │  / harness.py │                      工具适配器里一点
       │  清晰、可移植   │                      verifier 脚本里一点
       └──────────────┘                      运行时假设里一点
                                              ↑ 拼起来才是"有效 harness"
```

这个悖论本身就是一个**研究 gap** (你会在 L13 用 gap 雷达正式记下它): 我们有了统一理论, 却没有可移植的工程实现。

---

## 1. 为什么「不可移植」是个真问题

- **复现难**: 论文说「我们的 agent SOTA」, 但有效 harness 散落在 repo 各处 + 框架默认值里, 别人**复现不出**同样的分 (回忆 `critical-reading-gap` 的复现 gap)。
- **迁移难**: 想把一个 harness 从 OpenAI 换到 Anthropic、从一个任务搬到另一个, 得手动重接一堆散落的逻辑。
- **比较难**: 两个 agent 谁的 harness 更好? 没有「harness 作为独立工件」, 就难做公平对照 (L10 的评测困境根源之一)。

> 把 harness 提成可移植工件, 会同时改善**复现、迁移、评测**三件事——这就是为什么它是个值得做的研究方向。

---

## 2. 前沿一：portable / declarative harness

一个方向是让 harness **声明式 + 可移植**: 把 loop 策略、context 管理策略、工具集、控制规则, 写成一份**与具体框架/模型无关的规范**, 任何运行时都能加载执行。

- 这和 L03 的 provider 抽象是同一种精神, 升一个维度: 不只模型可替换, **整个 harness 可作为工件被搬运、版本化、比较**。
- 工程苗头: LangChain `deepagents` (把 write_todos 规划 / 文件系统卸载 context / subagent / 自动摘要打包成一个「全栈 harness」)、Claude Agent SDK (内建权限/hooks/跨会话桥接) ——它们都在朝「harness 作为可复用产品」走, 但「**完全可移植、框架无关**」仍是开放问题。

---

## 3. 前沿二：Natural-Language Agent Harnesses

更激进的方向 (arXiv 2603.25723, *Natural-Language Agent Harnesses*): **能不能用自然语言来表达 harness 逻辑本身?**

直觉: 既然模型擅长理解自然语言, 也许 harness 的部分控制逻辑 (何时停、何时换窗、何时调子 agent) 不必全写死成代码, 而可以用自然语言策略表达、由模型协同执行。

```
传统 harness:  控制逻辑 = 硬编码 (if budget>X: stop)
NL harness:    控制逻辑 = 自然语言策略 + 模型解释执行 (可读、可改、可学习)
```

- 诱惑: 更易读、更易改、甚至可被 agent 自己学习/改进 (回忆 Hashimoto: 出错就改环境——如果"环境"是自然语言, 改起来更快)。
- 风险/开放问题: 自然语言策略的**确定性、可验证性**怎么保证? (回忆 L02: harness 的价值正在于它是**确定性层**——把控制逻辑软化成自然语言, 会不会动摇这个根基?) 这正是有意思的研究张力。

---

## 4. 这一讲为什么放在 Part III 末尾

Part III 是「成熟度三件套」: 可观测 (L09) → 可评测 (L10) → **可移植 (L12)**。可移植是成熟度的最高阶: 一个领域的工件从「散落的隐性知识」变成「可搬运的显性工件」, 标志着它从手艺变成工程学科。**harness engineering 正处在这个转变的进行时**——这也是为什么现在入场做研究, 时机好。

---

## 5. 本讲小结 + 通往 Part IV

- 悖论: harness 能统一建模, 却散落各处、不可移植。
- 不可移植直接恶化复现/迁移/评测三件事。
- 两个前沿: portable/declarative harness (deepagents/Agent SDK 在路上) + NL harness (用自然语言表达控制逻辑, 但挑战确定性)。
- 可移植是成熟度最高阶; harness engineering 正在从手艺变工程学科。

> **Part IV 开始 (L13-L14)**: 你现在既懂工程, 也瞥见了好几处研究前沿。L13 把整门课里散落的 gap, 用你 `critical-reading-gap` 的 **6 类 gap 雷达**系统收集成候选研究题目; L14 Capstone 把全部 src 串成一个跑通长任务的升级版 harness, 并产出你自己的 idea 卡。**这就是「工程 → 研究」那座桥真正落地的地方。**
