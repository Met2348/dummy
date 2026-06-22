# L13 · 用 6 类 gap 雷达扫 harness 的开放问题

> Part IV · 40-min lecture · 衔接 `critical-reading-gap` (Module 9) · 目标: 把这门工程课变成研究起点——用你已经会的 6 类 gap 雷达, 把 harness engineering 里的开放问题, 系统收成候选研究题目。

---

## 0. 工程 → 研究的那一步

你现在懂了生产级 harness 怎么造。但**懂得造**和**做出新知识**是两件事 (这正是 Module 9 `critical-reading-gap` L3 讲的「消费 gap 的答案 vs 生产 gap 的答案」)。

这一讲把两门课接起来: 用 `critical-reading-gap` 的 **6 类 gap 雷达** (①方法 ②评测 ③假设 ④泛化 ⑤复现 ⑥理论), 对着 harness engineering 逐类扫, 产出真实的候选题目。**你不需要新领域, 这门课本身已经埋了一地 gap。**

> 复习 6 类 gap 与优先级 (来自 `critical-reading-gap/L3`):
> `Priority = Importance × Tractability / Cost`。博0 优先选 **可做性高、成本低** 的 (尤其 ⑤复现类)。

---

## 1. 逐类扫 harness（带优先级标注）

### ① 方法 gap — 「能不能做得更好/更省」
- **G1 更优的 compaction 策略**: 现在 5 阶段是工程直觉。能不能学一个策略, 决定「什么该 pin、什么该压、压多狠」, 在固定预算下最大化下游任务成功率? (来自 L04)
- **G2 更好的 completion-goal 判定**: loop-with-hook 靠 `goal_met` 判定, 可验证任务好办, 开放任务难。能不能学一个「任务是否真完成」的判别器? (来自 L05)
- 友好度: ★★ (需要训练/调优, Cost 中)

### ② 评测 gap — 「现在的测法对吗」
- **G3 harness 专用 benchmark**: 现有 benchmark (SWE-bench 等) 把模型和 harness 混在一起测。能不能设计一个**固定模型、只变 harness** 的标准评测协议? (来自 L10)
- **G4 长任务的过程评测**: 现在多看最终成功率。8 小时任务的**中间健康度** (context drift、state 退化) 怎么量化、怎么早期预警? (来自 L09)
- 友好度: ★★★ (评测类对博0 友好, Cost 可控)

### ③ 假设 gap — 「隐藏前提对吗」
- **G5 「文件系统是真相之源」的边界**: L05 假设 state 能干净落盘 + 读回。哪些任务的 state 无法良好序列化 (隐性上下文、连续控制)? 这个假设何时破裂? (来自 L05)
- **G6 「Model proposes, harness disposes」在 NL harness 下还成立吗**: L12 的 NL harness 把控制逻辑软化成自然语言, 动摇了「harness 是确定性层」这个根基假设。(来自 L02 + L12)
- 友好度: ★★ (假设类常出彩, 你 EE 的严谨是优势)

### ④ 泛化 gap — 「换场景还成立吗」
- **G7 harness 优化的跨模型泛化**: 一套对 Claude 调好的 harness (compaction 阈值/hook 策略), 换到 GPT/开源模型还最优吗? harness 超参有「模型脆性」吗? (来自 L03/L10)
- **G8 跨任务域泛化**: 为 SWE 调的 harness, 搬到科研/数据分析任务还行吗? (来自 L10)
- 友好度: ★★★ (系统性泛化研究, 可做)

### ⑤ 复现 gap — 「别人能重来吗」★ 对你最友好
- **G9 严肃复现 harness 论文的真实方差**: 很多 agent 论文 harness 散落、单配置、不报方差 (L12)。挑一篇热门的, **公平复现 + 公开代码 + 报告真实方差** (哪怕结论是「没那么强」)。
- **G10 「提升来自模型还是 harness」的解耦实验**: 系统地把若干 SOTA agent 论文的提升, 拆成「模型贡献 vs harness 贡献」。(来自 L10)
- 友好度: ★★★★ (你有 Module 7 的 harness 底子 + Module 9 的复现能力, 几乎现在就能做, Cost 低)

### ⑥ 理论 gap — 「为什么 work 不清楚」
- **G11 context folding 的形式化**: compaction 是有损压缩, 但「最小充分上下文」缺理论。能否给「在预算 B 下保留信息以最大化任务成功」一个形式刻画? (来自 L04/L12)
- **G12 long-horizon 失败的机理**: early-stopping / 跨窗口失忆 为什么发生? 是模型的什么性质导致的? (来自 L05)
- 友好度: ★★ (数学重, 你 EE 背景是优势)

---

## 2. 一张优先级速览（给博0 的起手建议）

| gap | 类型 | Importance | Tractability | Cost | 起手友好 |
|---|---|:--:|:--:|:--:|:--:|
| G9 复现+方差 | ⑤ | 中高 | 高 | 低 | ★★★★ |
| G10 模型/harness 解耦 | ⑤ | 高 | 高 | 中 | ★★★★ |
| G3 harness benchmark | ② | 高 | 中 | 中 | ★★★ |
| G7 跨模型泛化 | ④ | 中高 | 中 | 中 | ★★★ |
| G11 context folding 理论 | ⑥ | 高 | 低 | 低 | ★★ |

> 给你的判断 (同 `critical-reading-gap` L3 的建议): **第一个题目刻意选 ⑤复现类 (G9/G10)**——Tractability 高、Cost 低、且你有 Module 7 harness + Module 9 复现的双重底子。先完整跑通一次「读→找洞→做→写」的研究循环, 比一上来挑 G11 这种理论硬骨头更重要。

---

## 3. 本讲小结 + 通往 Capstone

- 用 6 类 gap 雷达扫 harness, 一门工程课就埋了 12 个候选题目。
- ⑤复现类 (G9/G10) 对你最友好, 是推荐的第一个题目。
- 每个 gap 都标了来源讲次——你可以回到那一讲深挖。

> **L14 Capstone**: 把全部 src (provider + compaction + long_horizon + otel + eval) 串成一个升级版 harness, 跑通一个跨窗口长任务; 然后**用 `critical-reading-gap/templates/idea-card.md`, 把上面 2-3 个 gap 正式写成 idea 卡**——带着它们去找 (准) 导师。这就是工程变研究的最后一步。
