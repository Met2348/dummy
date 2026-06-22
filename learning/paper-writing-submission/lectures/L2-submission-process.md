# L2 · 投稿全流程: venue 选择 / 时间线 / 格式匿名 / 提交机制

> 30-min lecture · 目标: 掌握投稿的完整流程地图 —— 选对会、卡准 deadline、过双盲匿名、用对 OpenReview/CMT。这是你以前没接触、但每篇论文都要走一遍的「工程」。

---

## 0. 写完论文只是一半

很多博0 以为「论文写完 = 大功告成」, 然后在投稿环节踩一堆坑: 投错会、错过 deadline、匿名泄露被 desk reject、格式不符被退回。**投稿是有流程、有规则、有坑的工程。** 这一讲给你地图。

---

## 1. venue 选择: 投哪里

NLP/LLM 的主战场是**会议**不是期刊 (CS 领域会议 > 期刊, 和你 EE 背景的直觉相反, 务必扭过来)。

### 主要会议日历 (NLP/ML, 记住大致节奏)

| 会议 | 领域 | 大致 deadline | 大致开会 |
|---|---|---|---|
| **ACL** | NLP 综合 | 2 月 (via ARR) | 夏 |
| **EMNLP** | NLP 综合 | 6 月 (via ARR) | 冬 |
| **NAACL** | NLP (北美) | 冬 | 春 |
| **NeurIPS** | ML 综合 | 5 月 | 冬 |
| **ICML** | ML 综合 | 1 月 | 夏 |
| **ICLR** | 表示学习/深度学习 | 9 月 | 春 |
| **COLM** | 语言模型专门 (新, 2024 起) | 春 | 秋 |

> 注: ACL/EMNLP/NAACL 现在多走 **ARR (ACL Rolling Review)** —— 一个滚动审稿池, 先投 ARR 拿 review, 再 commit 到某个会。日历每年微调, 以官网为准。**把这些 deadline 标进你 9.1 的日历**, 倒排研究节奏。

### 选会的判据 (匹配度 > 名气)

- **主题匹配**: 你的工作是 NLP 应用 → ACL/EMNLP; 偏方法/理论 → NeurIPS/ICML/ICLR; 纯 LM → COLM。投错社区, 再好也可能因「不 fit」被拒。
- **完成度匹配**: 完整工作 → 主会; 初步但有趣 → **workshop** (博0 的好起点! 门槛低、能拿反馈、能 networking, 见下)。
- **时间匹配**: 哪个 deadline 你能赶上且做扎实。**别为赶 deadline 投一个半成品** —— 拒稿浪费一轮 (几个月) 还打击信心。

> 给博0 的现实建议: **第一篇可以瞄 workshop 或 findings (EMNLP/ACL Findings)。** workshop 接收率高、反馈友好, 让你**先跑完一遍完整投稿循环**。回忆 9.3: 第一篇论文最大的价值是「证明你能跑完全程」, 不是中顶会。

---

## 2. 时间线: 从 deadline 倒排

deadline 不是一个点, 是一串:

```
   abstract ddl    full paper ddl    supplementary    (审稿期)    rebuttal期    通知
   (摘要先注册)  →  (正文截止)     →  (附录/代码)    →  ~6-8 周  →  ~1 周   →  录用/拒
   -7天            D-day            +几天                        ★L3        
```

倒排你的研究节奏 (现实化):
- **D-30**: 主结果跑完 (9.4/9.5), 别在最后一周还在调实验。
- **D-14**: 初稿完整 (用 9.7 装配 + 技能包写)。
- **D-7**: 图定稿 (9.6)、内部 review、改。
- **D-2**: 格式检查、匿名检查、附录、参考文献。
- **D-day**: 留 buffer 提交 (服务器临近 deadline 必崩, 别卡最后一小时)。

> 铁律: **永远不要卡最后一小时提交。** OpenReview/CMT 在 deadline 前几小时必然拥堵/崩溃, 每年都有人因此错过。提前 6-12 小时提交, 之后还能更新。

---

## 3. 格式与匿名: desk reject 的高发区

提交前的「形式审查」不过, 论文连审都不审就被退 (desk reject)。三个高频雷区:

### ① 格式合规
- 用**官方模板** (LaTeX style file), 别自己调。
- **页数限制**严格 (如正文 8 页)。超一行都可能被退。参考文献/附录通常不算页数, 看具体规定。
- 字号、边距别改 (有人为塞内容偷偷缩边距, 会被查)。

### ② 双盲匿名 (anonymization)
大多数会双盲 —— 审稿人不知道你是谁, 你不知道审稿人是谁。匿名要彻底:
- 正文不出现作者名、单位、致谢。
- **自引要中性化**: 别写「In our previous work [Zhang 2024], we...」(暴露身份), 写「Zhang et al. [2024] showed...」当第三方引。
- **别在论文里链接到你的非匿名 GitHub / 个人主页**。要放代码用**匿名仓库** (anonymous.4open.science)。
- 检查 PDF 元数据 (作者字段)、文件名别带你名字。

> 匿名泄露是**真的会被 desk reject** 的。投稿前用 anonymization checklist 过一遍 (本专题 `templates/submission-checklist.md`)。

### ③ 必交的声明
- **Limitations 节**: 很多会 (如 ACL) **强制**, 不算页数, 缺了直接拒。你 9.5 的不可复现点、9.4 的适用边界正好填这里。
- **Ethics / Broader Impact 声明**: 越来越多会要求。
- **Reproducibility checklist**: 提交时要勾 (你 9.5 的 checklist 直接对应)。

---

## 4. 提交机制: OpenReview / CMT / ARR

| 平台 | 谁用 | 特点 |
|---|---|---|
| **OpenReview** | ICLR / NeurIPS / COLM / 多数 | 公开 (或半公开) 评审, rebuttal 在平台 discussion 进行 (L3) |
| **CMT (Microsoft)** | ICML / 部分 | 传统封闭评审系统 |
| **ARR (ACL Rolling Review)** | ACL/EMNLP/NAACL | 滚动审稿, 先拿 review 再 commit 到会 |

提交时通常要填: 标题/摘要、作者 (匿名提交时平台知道但审稿人不知道)、主题领域 (primary/secondary area, 选准影响分到哪些审稿人)、利益冲突 (COI, 填你的合作者/导师以避开)、关键词。

> 提交 ≠ 不能改。多数平台在 full-paper deadline 前可反复更新版本。**先占位提交 (D-12h), 再继续打磨到 deadline。**

---

## 5. 本讲小结 + 通往 L3

- NLP/LLM 主战场是**会议**不是期刊 (扭过 EE 的直觉); 记住 ACL/EMNLP/NeurIPS/ICML/ICLR/COLM 日历。
- 选会看**匹配度 > 名气**; 博0 第一篇可瞄 **workshop / Findings**, 先跑完整循环。
- 从 deadline **倒排**研究节奏; **永不卡最后一小时**提交。
- desk reject 高发区: 格式 (模板/页数) / **双盲匿名** (自引中性化、匿名仓库) / 强制声明 (Limitations/Ethics)。
- 提交平台: OpenReview / CMT / ARR; 提交后 deadline 前可更新。

> **下一讲 L3「同行评审 + rebuttal」**: 提交后进入审稿。几周后你会收到审稿意见 (大概率有 negative)。怎么读它们、怎么写一份能翻盘的 rebuttal? 这是你以前完全没有、却决定论文生死的技能。

**动手**: 选一个你方向最对口的会 (如 EMNLP), 去它官网找到今年的 call for papers, 抄下: deadline、页数限制、是否双盲、是否强制 limitations。填进 `templates/submission-checklist.md`。
