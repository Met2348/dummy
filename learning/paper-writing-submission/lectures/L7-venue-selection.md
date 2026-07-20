# L7 · 会议 vs 期刊选择策略 + dual submission 规则

> 30-min lecture · 目标: `L2-submission-process.md` 已经教过"投哪个会"(ACL/EMNLP/NeurIPS 怎么选、workshop vs 主会怎么权衡)——本讲往上升一层, 讲一个 L2 完全没碰过的问题: **会议和期刊是两种根本不同的发表模型**, NLP/ML 领域还有介于两者之间的混合形态 (TACL/TMLR), 审稿风格和时间线差异很大; 并且讲清楚**双重投稿 (dual submission) 的红线到底划在哪、哪些是常见误解**。

---

## 0. L2 没告诉你的那一层

L2 帮你在"ACL 还是 EMNLP 还是 workshop"之间做选择, 这些全部属于**会议**这一种发表模型。但 CS/ML 领域 (和你的 EE 直觉不同, L2 已经提醒过一次: 这里会议 > 期刊) 其实还有别的选项: **传统期刊、以及 NLP/ML 领域特有的"期刊-会议混合体"**。选错发表模型, 代价和选错具体的会一样大, 甚至更隐蔽——因为很多博0 根本不知道混合模型的存在, 默认"论文只能投会"。

---

## 1. 会议 vs 期刊: 两种根本不同的审稿逻辑

先把两种模型的核心差异摆清楚, 这是理解后面一切的地基:

| | **会议 (Conference)** | **传统期刊 (Journal)** |
|---|---|---|
| 审稿轮次 | 通常**一次性** (single-shot): 提交→审稿→rebuttal (你唯一的申辩机会, L3) →录用或拒 | 可以有**多轮 revise-and-resubmit (R&R)**: 审稿人可以要求"改了再看一遍", 同一批审稿人跟到底 |
| 拒稿后果 | 拒了就要重新投一个**全新**的会 (从零开始, 换审稿人) | R&R 拒的是"这一稿", 不是"这个想法"——改完给同一批审稿人看, 不用重新排队 |
| 时间线 | 固定 deadline, 几个月内出结果 (L2 的时间线表) | 没有固定 deadline, 但单轮审稿+修改常常要半年到一年以上, 全程可能拖 1-2 年 |
| 篇幅/形式 | 严格页数限制 (L2 提过, 超一行都可能被退) | 通常篇幅更宽松, 允许更完整的方法论述和补充实验 |
| 领域惯例 | **NLP/ML 的主战场**, 影响力/引用集中在这里 | 在 CS 是次要战场, 但在你 EE 背景熟悉的信号处理/控制等领域仍是主战场 |

**这张表最重要的一行是"拒稿后果"**: 会议的 single-shot 逻辑意味着, 如果审稿人要求的修改**需要好几个月的新实验才能完成**, 你在会议的 rebuttal 窗口 (通常 1 周, L3) 里根本做不完, 大概率就是拒稿, 只能重新投一个新会、见新审稿人、故事从头讲起。这正是期刊 R&R 模型解决的问题——但传统期刊在 NLP/ML 领域声誉和曝光度都远不如顶会, 单纯"改投传统期刊"往往不是好选择。**这就是为什么 NLP/ML 领域发展出了下面这类混合形态。**

---

## 2. 混合形态: TACL 与 TMLR

### TACL (Transactions of the Association for Computational Linguistics)

TACL 是 ACL 官方主办、MIT Press 出版的期刊, 但发表**会议长度的论文**, 采用**期刊式审稿流程**——关键区别在于它保留了会议审稿没有的选项: **"revise and resubmit" (R&R, 拒稿但鼓励改后重投, 同一批审稿人继续跟审)**。更进一步, TACL 论文**可以选择在 ACL/EMNLP/NAACL 主会上做展示** (需满足特定时间窗口的资格条件), 相当于**用期刊的审稿深度, 换来会议的曝光舞台**——这在 NLP 领域是独一份的混合模型。

TACL 也明确规定了自己的**双重投稿红线** (下一节详谈): 任何提交给 TACL 的论文, 在审稿期间**不得**同时在其他期刊/会议/存档 workshop 接受审稿。

### TMLR (Transactions on Machine Learning Research)

TMLR 是 ML 领域较新 (2022 年创办) 的滚动投稿期刊, 核心理念和主会**审稿哲学完全不同**: **不以"新颖性/潜在影响力"作为录用标准, 只问两件事——① claim 有没有被准确、有说服力的证据支撑 ② 这项工作是否为社区贡献了知识。** 如果证据支撑不了某个 claim, 审稿人会要求你补证据或者**调低 claim 的强度**, 而不是直接因为"不够 novel"拒稿。TMLR 采用滚动投稿 (随时可投), 审稿周期短 (约 4 周出审稿意见, 2 个月出决定), 并且已经和 NeurIPS/ICML/ICLR 建立了 "Journal-to-Conference" 认证通道 (满足条件的 TMLR 论文可获邀在这些会议展示)。

> **TACL/TMLR 对你的实际意义**: 回忆 `L6-paper-series-strategy.md` 里 Robust-DPO 项目路线图的 Paper 2 (跨规模扩展性研究)——如果审稿人的核心质疑是"你需要在更多规模点上补实验才能验证这个 claim", 而这需要的算力和时间**远超会议 rebuttal 一周之内能做完的量**, 这正是该考虑投 TACL/TMLR 而非赌一次会议 single-shot 审稿的信号: 期刊式的 R&R 让你能在同一批审稿人面前**分阶段**把证据链补完整, 而不必因为"这次没来得及补"就整篇推倒重来、换一批新审稿人重新讲故事。

---

## 3. 审稿风格差异总表

把 L2 已讲的"会议内部怎么选"和本讲新增的"发表模型怎么选"合起来看:

| 发表模型 | 审稿轮次 | 单篇论文获得的关注度 | 适合的场景 |
|---|---|---|---|
| **主会 (ACL/EMNLP/NeurIPS 等)** | 单次, rebuttal 窗口短 | 3-4 个审稿人, 几周内速读 | 证据链已完整闭合、经得起一次性检验的成熟工作 (L1-L5 的 Robust-DPO 主线) |
| **Workshop (L2 已讲)** | 单次, 门槛更低 | 更少审稿人, 反馈更友好 | 博0 第一篇、初步但有趣的工作、L5 讲的负结果 (Insights workshop) |
| **TACL** | 可 R&R, 期刊式深度审稿 | 少数审稿人但跟审到底, 关注更深 | 需要补充大量实验才能让 claim 站住、且希望仍在会议舞台曝光的工作 |
| **TMLR** | 滚动投稿, claim-evidence 导向 | 关注"证据是否支撑 claim", 而非"是否够新颖" | 工作扎实但可能被主会嫌"不够 sexy/novel"、其实证据完全站得住的成果 |

---

## 4. Dual submission: 红线在哪

**dual submission (双重投稿)** 指的是: **把同一篇 (或实质相同的) 论文, 未经披露地同时提交给两个及以上的审稿流程, 让两边的审稿人同时花时间评审同一份工作。** 这是学术诚信红线 (呼应 L4 第 4 节), 但博0 常常对它的边界有误解。以 ACL Rolling Review (ARR) 的政策 (aclrollingreview.org 的作者指南 + ACL 官方 "Policies for Review and Citation") 为准, 讲清楚具体规则:

- **规则本身**: ARR 明确规定, 提交给 ARR 的论文**原则上不能同时在其他地方接受审稿** (除非有明确的截止日期重叠例外)。TACL 的规则更直白: 审稿期间, 论文的任何内容都不得同时在其他期刊/会议/存档 workshop 处于审稿状态。**核心是"同时被审"这个动作本身违规, 不是"最终发在哪"的问题。**
- **为什么这条线存在**: 每个 venue 的审稿人是义务劳动、资源有限。同一篇论文同时占用两个审稿池的人力, 是对审稿人劳动的浪费, 也让两边的编辑/领域主席无法基于"这篇论文只有我们在评审"的假设做出决定。

### 常见误解逐条澄清

- **误解① 「挂 arxiv 预印本 = 双重投稿」**: **不对。** ACL 政策自 2024 年 2 月起已明确: 审稿期间挂非匿名 arxiv 预印本, 不再有匿名期限限制, 也不构成 dual submission (回忆 L4 第 3 节 arxiv 策略)。dual submission 特指**同时提交给两个审稿流程**, 而不是"公开发布预印本"。这两件事在博0 中最常被混淆。
- **误解② 「投 workshop 和投主会可以同时进行, 因为 workshop '门槛低不算数'」**: **要看 workshop 类型。** 大多数 archival (存档制, 论文正式收录进 proceedings) workshop 和主会一样受 dual submission 规则约束。但也存在**明确合法的例外**: L5 提到的 Insights workshop 就设有"non-archival 摘要投稿"通道, 专门接收"已经发表或正在其他 venue 审稿中的工作", 这类投稿从设计上就**不是**去争夺一次原创审稿, 而是二次传播/讨论, 因此被明确允许。**区分关键在于 venue 是否把这类投稿设计为面向"已在他处发表/审稿"的工作**, 而非默认你在同时抢两次原创评审。
- **误解③ 「只要我在拿到决定前从一边撤稿, 就没有违规」**: **不成立。** 违规发生在"同时占用两个审稿池的义务劳动"这个行为本身, 而不是最终有没有拿到两个录用通知。哪怕你后来主动撤稿, 审稿人已经付出的时间已经被无谓消耗, 事后撤稿不能追溯抹除这个既成事实。ARR 政策里"authors may resubmit... must provide a link to the previous submission"这条要求 (改投必须挂钩前一次提交记录), 正是为了让系统能追踪"这篇稿子是不是已经在别处审过", 而不是纵容"先同时投、later 自己选一个"。
- **误解④ 「dual submission 和 L6 讲的 salami slicing 是一回事」**: **不是。** dual submission 是"同一篇论文同时被两个审稿流程审"; salami slicing 是"同一批研究被拆成多篇论文分别投出, 每篇只审一次, 但彼此实质雷同"。两者机制不同 (同时 vs 分批)、后果也不同, 但都属于"不当占用/重复利用审稿与发表资源"这个更大的伦理谱系——COPE 对"redundant publication (重复发表)"的界定同时覆盖这两种模式, 值得放在一起理解, 但处理和判断标准 (第 4 节 vs L6 第 4 节) 是两套。

---

## 5. 常见误区

| 误区 | 真相 |
|---|---|
| 「期刊在 CS/ML 领域没用, 只投会就够了」 | TACL/TMLR 这类混合形态期刊在特定场景 (需要 R&R、claim 需要分阶段补证据) 明显优于赌一次会议 single-shot 审稿; 完全不了解这个选项, 会在遇到"改不完的 rebuttal"时无路可走只能推倒重来 |
| 「TMLR 因为不看 novelty, 是灌水/兜底的地方」 | TMLR 的 claim-evidence 标准往往比"是否够新颖"更严格——它不接受任何证据撑不住的宣称, 只是把"重不重要"的主观判断留给了读者而非审稿人, 这和"审核宽松"是两回事 |
| 「只要没被两边同时录用, 同时投几个会试试运气没关系」 | 违规发生在同时审稿这个动作本身 (第4节误解③), 不取决于最终结果; 抱着"先投着看哪边先给结果"的心态本身就已经违规 |
| 「TACL/TMLR 流程比会议慢, 不适合任何着急发表的场景」 | 要具体看瓶颈在哪: 如果瓶颈是"审稿人要求补的实验做不完", 会议模式下你会因此整篇被拒、重新投一个新会耗时更久; TACL/TMLR 的 R&R 反而可能比"会议拒稿→重投新会→重新走一轮 review"更快让论文真正定稿 |

---

## 6. 参考

- **ACL Rolling Review (ARR), Authors Guidelines** (aclrollingreview.org/authors) 与 **ACL Policies for Review and Citation** (aclweb.org/adminwiki) —— dual submission 规则、preprint/arxiv 政策 (2024年2月起更新) 的官方依据, 本讲第 4 节的出处。
- **Transactions of the Association for Computational Linguistics, About the Journal / Submission Guidelines** (transacl.org, MIT Press) —— TACL 的 revise-and-resubmit 流程、会议展示资格条件、以及自身的 dual submission 政策, 本讲第 2 节 TACL 部分的出处。
- **Transactions on Machine Learning Research, jmlr.org/tmlr** —— TMLR 的 claim-evidence 审稿哲学、滚动投稿时间线、Journal-to-Conference 认证通道, 本讲第 2 节 TMLR 部分的出处。
- 也见本专题 `L2-submission-process.md` —— 本讲建立在 L2 已讲的会议内部选择 (ACL/EMNLP/NeurIPS/workshop、匹配度判据) 之上, 只补 L2 完全没涉及的"会议 vs 期刊"这一层与 dual submission 的深层规则, 两讲互补。
- 也见本专题 `L6-paper-series-strategy.md` —— 本讲第 2 节 TACL/TMLR 的适用场景直接用了 L6 走查示例里的 Paper 2 (跨规模扩展性研究) 做具体演示。

---

## 7. 本讲小结 + 通往 L8

- 会议是 **single-shot** 审稿模型 (rebuttal 是唯一申辩机会, 拒了要重新投新会); 传统期刊允许 **revise-and-resubmit**, 但在 NLP/ML 领域曝光和影响力不如会议。
- **TACL/TMLR** 是 NLP/ML 特有的混合形态: TACL 用期刊式 R&R 换会议舞台曝光; TMLR 用 claim-evidence (而非 novelty) 的审稿哲学、滚动投稿。当审稿人要求的修改超出会议 rebuttal 窗口能完成的量, 这是该考虑它们的信号。
- **dual submission 红线**: 同时把同一论文提交给两个审稿流程本身就违规, 不取决于最终结果; 挂 arxiv ≠ dual submission; 非 archival 的"二次传播"投稿通道是合法例外; 撤稿不能追溯抹除已发生的违规; 它和 L6 的 salami slicing 是两套不同机制。

> **下一讲 L8「camera-ready 与后续维护」**: 无论你的论文最终从哪个 venue 录用, 拿到 accept 通知之后, 到真正定稿提交之间还有一段容易被博0 低估的工作——L8 教你 camera-ready 阶段要做什么、artifact evaluation 怎么准备、代码数据什么时候放出来。

**动手**: 拿你自己 (或想象中) 某篇论文候选, 用第 3 节的表格判断: 它现在的证据链完整度, 更适合赌一次主会的 single-shot 审稿, 还是应该考虑 TACL/TMLR 这类允许 R&R 的模式? 说出具体理由 (证据链哪里可能不够、需不需要多轮修改空间)。
