# L06 · subagent 作为 context firewall + debate 模式

> Part II · 40-min lecture · 目标: 把 subagent 从「并行干活的小弟」重新理解为一个 **上下文防火墙**; 并讲清 multi-agent debate 为什么更可靠。

---

## 0. subagent 的真正用途不是并行，是隔离

新手以为 subagent = 多开几个 agent 并行加速。错。在 harness engineering 里, **subagent 首要是一个隔离机制 (isolation mechanism)**。

> 一个被反复引用的论断: **subagent 的 context firewall, 就是整个 multi-agent 层的核心。**

```
       主 agent (干净、聚焦的上下文)
        │  "去把这个子任务做了, 只告诉我结论"
        ▼
   ┌──────────── context firewall ────────────┐
   │  subagent: 自己的隔离上下文                  │
   │  - 读 50 个文件、试错 10 次、塞满自己的窗口    │
   │  - 这些"脏"上下文 全部留在防火墙内             │
   │  - 重建的权限上下文 (可更严/更松)             │
   └───────────────────┬───────────────────────┘
                       │ 只回传一段"结论" (而非全过程)
                       ▼
       主 agent 上下文 只增加一小段, 保持干净
```

**价值**: 一个子任务可能要读几十个文件、反复试错——这会产生大量「脏」上下文。如果在主线里做, 主 agent 的上下文很快被污染、被挤爆。**外包给 subagent, 脏活在防火墙内完成, 主线只收一段干净结论。** 这是给上下文减负的第三种方式 (前两种: L04 compaction、L05 换窗)。

---

## 1. firewall 的两道墙

```
subagent context firewall =  ① 上下文隔离   +   ② 权限重建
                             (脏上下文不外泄)    (rebuilt permission context)
```

### ① 上下文隔离
subagent 有自己独立的上下文窗口。它读了什么、试错了什么, 主 agent **看不到也不需要看到**。回传的只有最终产物 (一段总结、一个文件、一个结构化结果)。

### ② 权限重建（rebuilt permission context）
subagent 启动时**重新构建权限上下文**, 而不是继承主 agent 的。这意味着可以:
- **收紧**: 给一个「只读分析」subagent 去掉所有写/执行权限 (最小权限原则)。
- **放松**: 给一个受信任的子流程更高权限, 但范围被 firewall 框住。

> 这呼应 L08 的安全主题: **destructive-action hooks 坐在权限门后面, subagent firewall 是 multi-agent 层的全部, 工具派发注册表是 MCP 和 bash 共同的插入点**——三者构成 harness 的控制面。

---

## 2. 你已经有这块的基础

你 Module 7 的 `agent-harness-design/src/harness/subagents.py` 已经实现了 `run_subagent / fan_out` 的隔离派生。本专题不重复造, 而是补上**生产视角**:
- subagent 的**结果如何回传**才不破坏主线上下文 (只传结论 + 引用, 不传过程)。
- subagent 的**失败如何不拖垮主 agent** (firewall 也隔离失败)。
- subagent 的**权限如何重建** (而非继承)。

---

## 3. debate 模式：多 agent 为什么更可靠

一个值得记的多 agent 模式——**debate (辩论)**:

```
   提议者 (Proposer)  ──►  批评者 (Critic)  ──►  综合者 (Synthesizer)
   给出一个方案/答案        专门挑它的毛病         融合, 产出更稳的结论
```

> 实证: 对工程与分析类任务, **debate 模式 (一个提议、一个批评、一个综合) 比单 agent 生成更可靠。**

为什么有效, 用你 `critical-reading-gap` 的语言就一句话: **它把「攻击式阅读」内建进了生成过程**——提议者负责生成, 批评者扮演那个「想拒掉这篇」的审稿人, 综合者做策展。单 agent 容易被自己的第一个想法说服 (没人唱反调); debate 强制引入对抗。

注意成本: debate 要多次模型调用, 不是所有任务都值得。**它属于 L11 五大架构模式里, 你按任务可靠性需求挑选的一种**, 不是默认。

---

## 4. 本讲小结 + 通往 L07

- subagent 首要价值是**隔离**, 不是并行: 它是 context firewall。
- firewall 两道墙: 上下文隔离 + 权限重建。
- debate 模式 = 把攻击式阅读内建进生成, 对分析/工程类任务更可靠 (但更贵)。

> **下一讲 L07**: 工具与 MCP。subagent firewall 是 multi-agent 层, 而**工具派发注册表是 MCP server 和 bash 共同的插入点**。2026 出现了把 LLM gateway + MCP gateway + A2A gateway 合一的 **控制平面 (control plane)**——harness 的「神经中枢」长什么样。
