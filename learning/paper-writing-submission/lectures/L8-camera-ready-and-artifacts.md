# L8 · Camera-ready 与后续维护: accept 之后到定稿之间要做什么

> 25-min lecture · 目标: `L4-after-decision.md` 已经列过 camera-ready 阶段要做的事项清单 (改进/去匿名/加页/版权/代码数据/presentation)。本讲把这份清单**按时间顺序具体化**成一个可执行的操作时间表, 并深入两件 L4 只是一笔带过的事: **artifact evaluation (AE, 如果你的 venue 有) 具体要准备什么**, 以及**代码/数据发布的时间点到底该怎么定**——这两件事做不好, 会让一篇已经录用的论文在后续留下"说话不算数"的记录。

---

## 0. camera-ready 是一种新型 deadline, 和之前遇到的都不一样

L1-L7 你面对的 deadline 都是**研究 deadline**——距离越久, 你越能靠"多跑一组实验"补救。camera-ready deadline 完全不同: 它通常只有**录用通知后 1-3 周**, 而且它不是研究 deadline, 是**运营 (operations) deadline**——你不能靠"再跑个实验"解决问题, 只能靠**把已经答应的事逐项做完**。很多博0 第一次遇到 camera-ready 会犯的错, 是把它当成"论文已经过了, 可以放松"的信号, 结果在这个短窗口里手忙脚乱漏掉硬性要求 (漏掉某项声明、代码没放出来却在 checklist 里勾了"是")。**本讲把这个窗口拆解成一张可以照着做的时间表。**

---

## 1. camera-ready 窗口: 一张按天顺序的操作清单

假设你有 2 周窗口 (常见区间), 建议这样分配:

```
Day 0 (收到 accept 通知):
  · 先庆祝, 然后立刻通读一遍 meta-review + 所有 rebuttal 期承诺过的修改 (L3),
    列一张"必须兑现"清单——这是整个窗口最高优先级的任务, 逐条对照, 一个都不能漏。

Day 1-3: 兑现 rebuttal 承诺 + 去匿名
  · 把 rebuttal 里"我们会在正式版补充 X"的每一条落实到正文 (AC 真的会核对, L4 已强调)。
  · 去匿名: 加回作者/单位/致谢; 检查 PDF 元数据、文件名、脚注链接不再指向匿名仓库。
  · 如果 venue 给录用论文 +1 页 (常见), 优先用来补 rebuttal 期新增的内容, 而非塞新料。

Day 4-7: 代码/数据从匿名仓库过渡到正式仓库 (第3节详谈)
  · 补全 README、依赖清单、复现脚本; 这几天是最容易因赶deadline而"糊弄过去"的阶段,
    需要专门留出时间, 而不是留到最后一天。

Day 8-10: 若 venue 有 artifact evaluation / reproducibility checklist (第2节详谈)
  · 按对应标准打包、写复现说明, 提前预演一遍"如果我是审核者, 这份包能跑通吗"。

Day 11-13: 格式/声明/版权收尾
  · Limitations/Ethics 声明终稿检查、copyright form 签署、参考文献格式核对。

Day 14 (留 buffer): 提交, 别卡最后一刻——和 L2 的投稿铁律一样, camera-ready 系统临近
  deadline 同样会拥堵。
```

> 这张表的核心原则: **把"兑现承诺"排在最优先, 而不是留到最后。** 承诺没兑现 (rebuttal 里答应的修改、checklist 里勾选的代码开源) 是学术诚信问题 (呼应 L4 第 4 节), 一旦被 AC 或读者发现"说了会做但没做", 伤害的是你长期的可信度。

---

## 2. Artifact Evaluation (AE): 如果你的 venue 有

### 什么是 AE, 通用标准长什么样

**Artifact Evaluation (人工制品评审)** 是一部分会议 (更常见于系统/软件工程类, 如 MLSys、VLDB, NLP 主会目前较少有独立的正式 AE track) 在论文录用**之后**单独设立的环节: 由一组评审员实际尝试运行你提交的代码/数据, 检验它是否真的能重现论文里的结果。ACM 官方的 **"Artifact Review and Badging"** 政策 (acm.org 发布, 被 SIGIR/MMSys 等多个 ACM 系会议采纳) 定义了标准化的三类徽章:

| 徽章 | 含义 |
|---|---|
| **Artifacts Available** | 制品被永久性公开存档 (不要求已被评审或完整, 只要求真实可获取且和论文相关) |
| **Artifacts Evaluated (Functional / Reusable)** | 制品文档完备、可运行; 更高一级要求结构清晰到"别人能直接复用" |
| **Results Validated (Reproduced / Replicated)** | 论文核心结果被作者以外的人独立复现 (Reproduced = 用作者提供的制品复现; Replicated = 不借助作者制品、独立复现) |

一个重要的**流程保护**: 大多数采纳这套政策的会议明确规定, **AE 评审在论文已经被接收之后才开始**, 结果**不影响录用决定** (参不参加 AE、AE 结果如何都不倒推影响 accept/reject)——这是为了让 AE 保持"锦上添花"的定位, 而不是变成变相的二次审稿关卡。

### NLP/ML 领域的对应物: Responsible NLP Research Checklist 与 ML Reproducibility Challenge

NLP 主会 (ACL/EMNLP/NAACL, 经 ARR) 目前**没有**独立的 ACM 式 AE track, 但有它自己的等价机制: **ARR 的 "Responsible NLP Research Checklist"**——这份 checklist 从 2023 年起被整合进投稿表单本身 (而非事后附加环节), 要求作者在**投稿阶段**就申报是否会公开代码、数据许可证、计算资源消耗等信息, 相当于把"是否可复现"提前变成录用过程的一部分, 而不是事后再补。**这份 checklist 里你申报"会公开代码"这一项, 到了 camera-ready 阶段就是你必须兑现的承诺**——回到第1节: 这正是为什么"代码/数据发布"要排进 camera-ready 时间表, 而不是随口一说。

另一个值得知道的社区活动是 **ML Reproducibility Challenge (MLRC)**——一个从 2018 年 ICLR 起步、现已覆盖 NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR/ECCV 等七大顶会的社区复现挑战: **任何人** (常是其他实验室的研究生) 可以挑一篇你发表的论文, 尝试复现它, 并把复现报告投给 MLRC/TMLR。这不是你必须参加的强制流程, 但它意味着: **你的论文发表之后, 随时可能有陌生人真的去跑你放出的代码。** 这是"代码/数据发布质量"最终极的检验——不是走个过场就完事。

### 如果你的 venue 真的有正式 AE track, 具体要准备什么

- 一个能**从零环境**跑起来的复现脚本 (pin 住依赖版本, 最好提供容器化方案), 而不是假设评审员的环境和你一模一样。
- 一份 `REPRODUCING.md`, 写清楚: 预期跑多久、需要什么硬件 (GPU 型号/显存)、跑完应该看到什么数字 (和论文里报告的数字允许多大误差, 呼应 9.5 experiment-ops-repro 的"留痕"精神——种子/版本/硬件都要写明)。
- 提前**自己用一台干净的机器/新建的虚拟环境跑一遍你打包的东西**——这是最容易被忽略但最有效的自查, 很多"能跑"的代码其实偷偷依赖了你本地环境里一个没写进依赖清单的包。

---

## 3. 代码/数据发布: 什么时候放出来

第 2 节已经点出核心张力: **你在 checklist/rebuttal 里承诺的开源, 必须有一个明确的兑现时间点。** 三个选项各有场景:

| 时间点 | 适合场景 | 权衡 |
|---|---|---|
| **投稿时就放 (匿名版本)** | 想让审稿人在评审阶段就能验证你的结果, 提高说服力 (回忆 L3: 补实验/证据是最强的翻盘武器, 可复现的代码同样加分) | 必须做好匿名处理 (L2 第3节: 用匿名托管服务, 不能暴露身份信息) |
| **camera-ready 时放 (默认选项)** | 最常见的选择——和"去匿名"同一批动作一起做, 时间点清晰、责任明确 | 需要在第1节的 Day 4-7 专门留出打磨代码的时间, 不能挪到最后一天糊弄 |
| **camera-ready 之后、有明确日期的延后** | 代码依赖内部工具/许可证审批, 短期内确实放不出来完整版 | **必须在论文里给出一个具体承诺** (如"完整代码将于 X 月发布"), 不能只写"代码将会发布"这种没有时间约束的空话——没有具体日期的承诺等于没有承诺, 后续容易不了了之被读者质疑 |

> 一条容易被忽视的原则 (呼应 NeurIPS 2019 reproducibility program 的实际经验): **代码"能跑但杂乱"好过"赶deadline前完全不放", 但"完全不能跑、文档缺失"的代码比"晚几天但完整可用"更伤害你的可信度。** 如果 Day 4-7 发现代码还没整理好, 优先选择"承诺一个具体的补发日期" (并且真的做到), 而不是硬塞一份跑不通的代码去应付 checklist。

---

## 4. 常见误区

| 误区 | 真相 |
|---|---|
| 「录用了, camera-ready 就是走个形式, 随便填填」 | 这是一个运营 deadline, 不是形式——checklist 里勾的"是", AC/读者/MLRC 的陌生复现者都可能真的去核对, 说话不算数的代价比想象的大 |
| 「AE / reproducibility checklist 是可选的, 不参加没关系」 | ARR 的 Responsible NLP Research Checklist 已经整合进主流程本身, 不是"可选加分项"; ACM 式独立 AE track 在你参加的具体 venue 才是真正可选 (先查清楚你的 venue 属于哪种) |
| 「代码要么现在放, 要么永远不放, 没有中间态」 | 第3节的第三选项 (给出具体日期的延后发布) 是被广泛接受的诚实做法, 前提是日期要具体且真的兑现 |
| 「AE 评审结果不好会影响我这篇论文的录用状态」 | 主流政策 (ACM 式 AE) 明确规定评审在录用之后才开始, 不倒推影响 accept/reject——它是给你和读者的额外信息, 不是二次审稿 |

---

## 5. 参考

- **Pineau, J., et al., "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)", *Journal of Machine Learning Research*, 2021** —— NeurIPS 代码提交政策 (强调"仅对录用论文, 仅要求到 camera-ready deadline 前")的官方报告, 本讲第0/3节的出处。
- **ACM, "Artifact Review and Badging" (Current version)**, acm.org/publications/policies/artifact-review-and-badging-current —— Available/Evaluated/Results Validated 三类徽章的标准定义, 本讲第 2 节 AE 通用标准的出处。
- **ACL Rolling Review, "Responsible NLP Research Checklist"**, aclrollingreview.org/responsibleNLPresearch —— NLP 主会自己的可复现性申报机制, 本讲第 2 节的出处。
- **ML Reproducibility Challenge (MLRC)**, reproml.org —— 覆盖 NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR/ECCV 的社区复现挑战, 本讲第 2 节"陌生人真的会跑你的代码"这一现实依据。
- 也见本仓库 `experiment-ops-repro/README.md` (9.5) —— 该专题教你在**执行阶段**就把可复现性做对 (固定 seed/记录 git sha/超参管理); 本讲讲的是**发表之后**怎么把这套东西打包交付给外部审核者/复现者, 两者是同一条可复现性主线的前后两段。
- 也见本专题 `L4-after-decision.md` —— 本讲是 L4 camera-ready 清单的时间线具体化 + AE/代码发布两个专项深挖, 不重复 L4 已讲的去匿名/版权/学术诚信红线内容。

---

## 6. 本讲小结 + 通往 L9

- camera-ready 是**运营 deadline**, 核心任务是**兑现承诺** (rebuttal 修改、checklist 勾选项), 建议按第 1 节的按天时间表分配任务, 别留到最后一天。
- **AE (如果 venue 有)**: ACM 式独立 AE track (Available/Evaluated/Results Validated 三级徽章) 在评审录用之后进行、不影响录用结果; NLP 领域更常见的等价物是已整合进投稿流程的 Responsible NLP Research Checklist, 加上社区自发的 ML Reproducibility Challenge 作为"陌生人真的会来跑你代码"的现实检验。
- **代码/数据发布时间点**: 投稿时(需匿名)/camera-ready时(默认)/有明确日期的延后, 三选一都可以, 但"没有具体日期的空头承诺"和"硬塞跑不通的代码"都不可取。

> **下一讲 L9「论文的长期影响力经营」**: 代码放出来、论文定稿了, 只是这篇论文生命周期的开始, 不是结束。L9 教你怎么让论文被真正相关的人看到 (而不是灌水式刷引用), 以及怎么规划它的 follow-up。

**动手**: 假设你手头有一篇刚收到 accept 通知的论文 (或用 Robust-DPO 例子), 按第 1 节的时间表列出你自己 2 周窗口内的具体任务清单, 并针对"代码发布"明确写下你会选第3节哪个时间点、为什么。
