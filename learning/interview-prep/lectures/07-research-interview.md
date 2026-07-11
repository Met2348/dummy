# L07 · 研究面试:论文 drill + 开放式推理 + hero project

这是**你的主场**(70% 权重)。研究岗/PhD 暑研的核心不是 coding,是"你会不会做研究"。

## 三件事

### 1. 论文口头 drill 甲板

选**你方向的 10 经典 + 10 前沿**(judge-internals / RLHF / 可解释性)。每篇练到能上白板讲:
- **一句话动机**(什么问题、为何重要)
- **方法核心**(一张图能画出来)
- **关键结果 + 局限**(不吹,能说出它哪里不成立)
- **一个 follow-up**(你会怎么改进/推进)

> 面试官常递一篇你没读过的论文让你现场读+批判。练的是**读得快、抓得准、有观点**,不是背诵。

### 2. 开放式科研推理

典型问法:
- "你会怎么验证 X?"→ 给出可证伪假设 + 实验设计 + baseline + 混淆控制。
- "这个结果,什么会 confound 它?"→ 展示批判性。
- "如果给你无限算力/数据,你研究什么?"→ 展示品味。

用 Module 9 的 `experiment-design` 框架:假设 → 变量 → 对照 → 可证伪 → 预注册。

### 3. Hero project(面试弹药 × 博士启动,同一件事)

自学 notebook 难讲 impact。你需要 **1 个真复现**:真模型(GPT-2/Pythia)、真跑 probing / activation patching / logit lens / 小 SAE、**真数字 + 公开 repo**。

**关键:对接你已有的 PhD 仓**(70-paper repo + 24-src audit + judge-internals 方向),不另起炉灶。详见 `docs/superpowers/plans/2026-07-11-interview-research-hero-project.md`。

面试里它同时是:
- **behavioral 素材**:"我做了 X,发现 Y,证明了 Z"。
- **研究品味展示**:讲清你为什么这么设计、什么反驳过你、下一步。
- **PhD 开局**:直接喂进你第一年的工作。

## 为什么这比再刷题重要

研究岗录用信号里,**"能独立推进一个真问题并讲清楚"** 远重于"LeetCode 手速"。coding 只要过地板;研究深度决定 offer。所以 70% 权重在这。
