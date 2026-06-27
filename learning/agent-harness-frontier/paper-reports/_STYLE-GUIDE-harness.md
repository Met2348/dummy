# 组会汇报文档 · Agent-Harness 库附加规范（harness 专属增量，只加不减）

> 本库 = **agent harness**（把 LLM 变成 agent 的软件层）。报告写作**完全沿用** auto-research 库已验证的两份规范：
> - **v1 硬规范**：[`../../auto-research-frontier/paper-reports/_STYLE-GUIDE.md`](../../auto-research-frontier/paper-reports/_STYLE-GUIDE.md)
>   （中文+术语对照、每公式前给直觉+先定义符号、setting/metrics/params 全、指标给定义式、数字标 §/Table/Eq 出处、
>   忠于原文、PPT 风格、约 20 页、20 页骨架）——**100% 生效**。
> - **v2 增量**：[`../../auto-research-frontier/paper-reports/_STYLE-GUIDE-v2-why-and-inspiration.md`](../../auto-research-frontier/paper-reports/_STYLE-GUIDE-v2-why-and-inspiration.md)
>   （**Why 三连**：问题层/设计层/结果层 + 强制 `## ★ 对我们的启发（Inspires Us）` 一节）——**100% 生效**。
>
> 本文件在以上之上，再加 5 条 **harness 专属**要求（Θ1–Θ5）。**写每篇前先读完上面两份 + 本文件。**

---

## Θ1. 标注 harness 分层：E/T/C/L/O/V
本库用六层给 harness 解剖：**E**nvironment 环境 / **T**ools 工具 / **C**ontext 上下文 / **L**oop 控制循环 /
**O**bservability 可观测 / **V**alidation 验证。每篇报告须在 **§1 TL;DR** 与 **§17 版图定位**点明"这篇主要打哪一层、
对其它层有何依赖"。库内分组与之对应：A 综述（跨层）/ B=L / C=T / D=C / E=集成系统 / F=Environment / G=V / H=O。

## Θ2. 回扣全库论点：**Agent = Model + Harness**
§17 必须回答："这篇为『harness 决定能力』这一命题贡献了什么证据 / 工具 / 反例？" 凡论文给了
**同模型换 harness 的数字摆动**（如 scaffold 42%→78%、Cursor 46%→80%、Vercel 砍工具 80%→100%、
Anthropic 系统卡 harness 7 分差），务必抓出来，按 v1 的"指标定义式 + 机制解释"写清，不要只复述数字。

## Θ3. Inspires-Us 要"打到自己身上"——我们就活在一个 harness 里
v2 的 `## ★ 对我们的启发` 在本库有天然优势：**我们（Claude Code / 本课 m9.* 的 agent）本身就是一个 harness**——
有真实的 ReAct 循环、工具预算、上下文压缩 / compaction、子代理编排。所以本节的 **e（我的下一步）必须落到
我们自己 harness 的具体组件改动**——"我会把这篇的 X 加进我们的〔工具层 / 上下文压缩策略 / 控制循环 / 子代理分工〕，
测 Y 是否改善"，而非泛泛感想。这是本库相对 auto-research 最大的"第一人称"优势。

## Θ4. canon vs 前沿：诚实标定时间坐标
本库含约 30 篇 2022–2024 基石。§17 须诚实标定：
- 若写的是 **canon**（ReAct / Reflexion / MemGPT / SWE-agent / Toolformer / WebArena…）：写清"它定义/奠基了什么、后续谁在它上面长肉"。
- 若写的是 **2025–2026 前沿**：写清"它相对基石推进了哪一步、或证伪 / 收紧了谁"。

## Θ5. regime 诚实：别把"harness > model"写成绝对真理
"harness 比 model 重要"是**分 regime** 的：Harvey / CORE-Agent / Cursor 一侧 harness 增益巨大；
但 METR / Scale AI(SWE-Atlas) 一侧发现 harness 选择在误差范围内、强模型可能反而降低对脚手架的依赖。
报告须**区分论文宣称 vs 独立反证**，给出"什么条件下 harness 主导、什么条件下不"。这是本库的判断力护城河
（对应 auto-research 的 G 组批判）。

---

## 自检（在 v1 + v2 自检之外**追加**）
- [ ] §1 与 §17 是否标了 **E/T/C/L/O/V** 层归属？
- [ ] §17 是否回扣 **"Agent = Model + Harness"**，并抓出 harness 数字摆动（若论文有）？
- [ ] Inspires-Us 的"下一步"是否落到**我们自己 harness 的具体组件**（工具/上下文/循环/子代理）？
- [ ] 是否诚实标了 **canon / 前沿**坐标 + **regime 依赖**（不把 harness>model 绝对化）？
- [ ] v1 + v2 的全部硬性要求是否仍 100% 满足？
