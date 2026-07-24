# 02 · Introduction 段落级写作与 Related Work 定位

> 总览见 [00-roadmap.md](00-roadmap.md)。上一篇([01](01-narrative-structure-and-elevator-pitch.md))
> 讲的是整篇论文的故事线形状,这一篇下沉到 Introduction 每一段该承担什么功能、Related Work 该怎么
> 站在前人肩膀上定位贡献,而不是"列一堆引用了事"。

---

## 1. Introduction 只做两件事:描述问题 + 声明贡献

SPJ 的建议是把 Introduction 严格控制在一页以内,只做两件事:**①描述问题,②声明贡献**,不做别的。
描述问题时,他明确反对用抽象泛化开场,举了一个他自己嘲讽的反例:"Computer programs often have bugs.
It is very important to eliminate these bugs [1,2]. Many researchers have tried [3,4,5,6]. It
really is very important."——这种开场每一句都对,但没有一句提供具体信息,建议改用一个具体例子把问题
带出来。声明贡献时,他强调**先写贡献列表、用 bullet point 列出来、每条都可证伪**,不要留给读者自己去
Introduction 正文里"猜"贡献是什么。他还特别反对结尾加一段"The rest of this paper is organized as
follows: Section 2 introduces...",认为这种"路线图段落"没有信息量——更好的做法是让 Introduction 本身
(包括贡献列表)已经对全文做了一次前瞻式的概览,靠正文叙事自然带出后续章节,不需要额外一段公式化的
目录复述。

**常见误区/反面例子:**

> Reinforcement learning has achieved great success in recent years [1,2,3]. World models are
> an important component of many RL systems [4,5]. However, there are still many challenges
> in this area. In this paper, we address one of these challenges. The rest of this paper is
> organized as follows: Section 2 discusses related work, Section 3 describes our method,
> Section 4 presents experiments, and Section 5 concludes the paper.

四句话背景 + 一句"我们解决了其中一个挑战"(没说是哪个)+ 一段路线图。读者读完整段,连"具体解决了什么
问题"都不知道,更不用说"解决得怎么样"。

**逐处修改对照:**

| 原句 | 问题 | 改法 |
|---|---|---|
| "Reinforcement learning has achieved great success... World models are an important
  component... there are still many challenges." | 三层空洞背景,没有一句具体 | 换成一个具体场景:
  "一个 agent 在决策前用 world model 想象几步未来——但现有系统不管这次决策难不难,每次都花掉同样多的
  想象预算。" |
| "In this paper, we address one of these challenges." | 没说清楚解决了哪个挑战、怎么解决的 | 换成
  可证伪的贡献列表(见下方) |
| "The rest of this paper is organized as follows..." | 公式化路线图,零信息量 | 删除,让贡献列表本身
  完成"预告全文"的功能 |

改写后(贡献列表按 SPJ 建议用 bullet,每条可证伪):

> An agent that plans with a world model decides, before acting, how many future scenarios to
> imagine and how deep to imagine them. Existing systems fix this budget at training time and
> never revisit it per decision. This paper makes three contributions:
> - We show, with a controlled synthetic environment, that spending more computation on
>   imagination does not improve decisions when the rollout and the baseline policy share
>   exactly the same model (Section 3).
> - We show that giving the rollout genuine task-relevant information reverses this result,
>   raising the hit rate from 63.7% to 82.0% across three target settings (Section 4).
> - We connect both findings to classical value-of-computation theory and propose a controller
>   that allocates budget based on estimated information advantage (Section 5).

**可操作检查清单:**
- [ ] Introduction 第一段是否用一个具体场景/例子带出问题,而不是三句抽象背景堆叠
- [ ] 贡献列表是否用 bullet point 单独列出,而不是散落在段落叙述里让读者自己抠出来
- [ ] 每条贡献是否可证伪——读者读完这条,能不能想象出"如果这不成立,应该看到什么"
- [ ] 是否删除了"The rest of this paper is organized as follows"这类路线图段落,改由贡献列表的
  每条末尾自然标注对应章节号(如上面例子里的"(Section 3)")

**量化验证:** "问题描述是否够具体""贡献是否可证伪"是判断力问题,但"是否存在路线图套话段落"是可以
用字符串匹配可靠检测的表层特征——这个检查足够简单以至于不需要独立成一段代码,直接给出判断规则:
一段文字里如果包含 `"rest of this paper is organized"` / `"rest of the paper is structured"` 这类
固定短语,几乎可以确定是路线图段落,建议删除或者改写成贡献列表的自然延伸,这条规则会在
[04 类](04-sentence-level-academic-english.md)的套话检测器代码里作为通用套话清单的一员一起验证,
这里不重复贴代码。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"你的贡献列表第一条'我们证明了 X 在同源模型下不会变好',这条本质是负结果,
  凭什么算贡献?" → 反驳:负结果只要满足"可证伪 + 有受控实验支撑",就是合法贡献,SPJ 的可证伪标准
  没有区分"正面结果"和"负面结果"——真正的问题不是"这是不是负结果",而是它是否已经是文献里的已知
  结论(这个追问会在知识点 4、05 类、07 类反复出现,因为它是这个真实项目故事里最核心的一次教训)。
- **决策依据追问轴**:"为什么不写路线图段落,读者怎么知道整篇论文的结构?" → 反驳:结构应该靠贡献
  列表 + 正文小标题自然传达,而不是额外用一段"Section 2 讨论 X,Section 3 讨论 Y"的公式化文字——
  如果读者读完贡献列表还是不知道整篇论文的结构,说明问题出在贡献列表写得不够清楚,不是缺了路线图段落。

**常见坑:**
1. 贡献列表写成"我们提出了一个系统,它很有效"这种不可证伪的说法——和知识点重复的错误,但在 Introduction
   这个位置更致命,因为这是审稿人对整篇论文的第一印象。
2. 具体例子选得过于简单以至于无法体现问题的困难之处——好的开场例子应该恰好卡在"直觉上好像可以简单
   解决,但仔细想又不行"这个点上,才能自然带出"为什么这个问题值得做"。

---

## 2. Related Work 的"列一堆引用了事"陷阱

Related Work 最常见的失败模式是**一段一篇论文的罗列**——每段总结一篇论文,段落之间没有关联,读者
读完知道"有哪些论文",但不知道"这些论文之间是什么关系、共同缺了什么"。更好的做法是**按主题/技术路线
分组**,组内多篇论文放进同一段综合讨论,用一句话讲清楚"这条技术路线是什么、怎么做的、局限在哪",
而不是逐篇复述。这条建议在多个独立来源里高度一致:按子主题分组、组间综合而不是逐篇罗列、把"总结每篇
论文"这件事留在自己的读书笔记里,不要直接誊抄进正文。

**常见误区/反面例子:**

> [3] proposed a gating network that decides whether to call the world model. [7] introduced
> a verifier that checks whether a generated rollout can still be trusted. [12] studied search
> over generated video frames using a tree-structured verifier.

三句话三篇论文,句式几乎一样("[X] proposed/introduced/studied..."),读完不知道这三篇论文之间是并列
关系、递进关系还是竞争关系。

**逐处修改对照:**

改写后(按"处理的是决策链条上哪个环节"这条主线组织,而不是按论文发表顺序或字母序罗列):

> Prior work on test-time imagination control splits along which decision point it targets.
> One line asks whether to imagine at all, using a binary gate trained with RL [3]. A second
> line asks, once imagining has started, whether the result can still be trusted [7]. A third
> treats imagination as a search problem, branching and pruning over candidate futures scored
> by a learned verifier [12]. None of the three compares a candidate future against other
> semantically different candidates using a decision-value objective, which is exactly the gap
> we target.

改动的关键不是句子变长了,而是**多篇引用被组织进了同一条叙事线**,并且结尾一句话直接点出这三条线
共同的缺口——这句收尾正是知识点 4 要展开的"Positioning"写法。

**可操作检查清单:**
- [ ] Related Work 是否按主题/技术路线分组,而不是按论文发表时间或字母序罗列
- [ ] 单篇论文单独成段的比例是否过高——如果大多数段落只讨论一篇论文,大概率落入了"一段一篇"陷阱
- [ ] 段落之间是否有明确的过渡语("A second line...","In contrast...","Building on this..."),让
  读者看得出论文之间的关系,而不是几段互不相关的摘要拼接
- [ ] 是否把"总结每篇论文在做什么"和"批判性指出局限"分开处理——先说清楚这条线**做出了什么**、
  **怎么做到的**,再指出局限,不要一上来就否定,"先教后评"比"直接建立自己的优越感"更能体现学术诚实

**量化验证:** "组织是否合理"是判断力问题,但"一段是否只引用了一篇论文"是可以用正则统计的表层信号——
下面的检查器按空行切分段落,统计每段引用的不同文献编号数,给出"单引用段落"的比例:

```python
import re

def paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def citation_keys(p):
    return set(re.findall(r"\[(\d+)\]", p))

def listing_trap_score(text):
    paras = paragraphs(text)
    per_para_keys = [citation_keys(p) for p in paras]
    single_cite_paras = sum(1 for keys in per_para_keys if len(keys) <= 1)
    return {
        "n_paragraphs": len(paras),
        "single_citation_paragraphs": single_cite_paras,
        "ratio_single_citation": single_cite_paras / max(len(paras), 1),
    }

listing_style = (
    "[3] proposed a gating network that decides whether to call the world model.\n\n"
    "[7] introduced a verifier that checks whether a generated rollout can still be trusted.\n\n"
    "[12] studied search over generated video frames using a tree-structured verifier."
)

synthesized_style = (
    "A first line of work asks whether to imagine at all, using a binary gate trained with "
    "RL [3]. A second line asks, once imagining has started, whether the result can still be "
    "trusted [7]. A third treats imagination as a search problem, branching and pruning over "
    "candidate futures scored by a learned verifier [12]. None of the three compares a "
    "candidate future against other semantically different candidates using a decision-value "
    "objective, which is exactly the gap we target."
)

r1 = listing_trap_score(listing_style)
r2 = listing_trap_score(synthesized_style)
assert r1["ratio_single_citation"] == 1.0
assert r2["ratio_single_citation"] == 0.0
print("listing_style (bad example):", r1)
print("synthesized_style (revised):", r2)
```

本机实测:`listing_style`(三段各自独立引用一篇文献)`ratio_single_citation=1.0`;`synthesized_style`
(合并成一段、三篇文献在同一段里被综合讨论)`ratio_single_citation=0.0`。**明确边界**:这个比例不是
"越低越好"的绝对指标——有些文献确实需要单独用一段深入讨论(比如和自己工作关系最紧密的一篇最近邻
工作),该检查器只能提示"大部分段落都是单引用"这个可能有问题的整体模式,不能逐段判断"这一段该不该
单独成段"。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"你把 [3] [7] [12] 归成'三条线',这个分类标准是你自己定的还是这个领域公认的?"
  → 反驳:分类标准需要在正文里明说依据是什么(这里的依据是"处理决策链条上的哪个环节"),不能只是
  "看起来能分成三类"就直接下结论——分类标准本身如果站不住,后面"没有一篇做到 X"的定位论证也会跟着
  垮掉。
- **真实性验证轴**:"你说'没有一篇比较语义不同的候选未来',查过多少篇相关论文才敢下这个结论?" →
  反驳:诚实的做法是标注调研范围和方法(比如"穷尽检索了 2024-2026 年 N 篇高度相关论文"),而不是
  含糊地说"据我所知"——范围越具体,这句结论越经得起审稿人自己去核查。

**常见坑:**
1. 只在 Related Work 一节做分组,但分组逻辑和后面 Method 部分讲自己方法时用的框架完全不搭——好的
   分组标准应该恰好能延伸出"我们的方法在这个分类框架下处于什么位置",两节要能接得上。
2. 为了追求"综合评述"的写法,把原本清晰的技术细节写得含糊——综合不等于模糊,合并讨论多篇论文时,
   每篇论文具体做了什么仍然要交代清楚,只是不再一段一篇地重复相似的句式。

---

## 3. Related Work 放在哪里:SPJ 的建议和一个公开的反对意见

SPJ 七条建议之一是"put related work at the end"——把 Related Work 放在方法和实验之后、结论之前,
理由是让读者先完整理解"你的想法",再去对比"别人的想法",不要还没讲清楚自己的贡献就先陷入和一堆前人
工作的比较。但这条建议本身是**有公开争议的**:有评论明确反对,认为 Related Work 至少还承担两个 SPJ
没有充分强调的功能——"论证这个问题有人关心"和"论证这个问题此前没被解决",这两者本质上是**motivation
的一部分**,应该在 Introduction 附近尽早出现,而不是拖到最后才讲;这条反对意见给出的替代结构是
Introduction(这是一个问题)→ Related Work(别人怎么处理过、为什么没有真正解决)→ 再进入方法细节。

**这一条知识点的教学意义,恰恰在于它没有一个"标准答案"**——两种结构在不同论文里都能找到成功案例,
选哪一种取决于具体场景:

**可操作检查清单(判断力清单,不是二选一的规则):**
- [ ] 如果不了解 Related Work 就无法理解"我的方法为什么长这样"(比如方法是针对某个前人方法的具体
  局限做的改进),倾向于把 Related Work 提前,放在方法之前起到 motivation 的作用
- [ ] 如果自己的方法可以独立于 Related Work 被理解(不依赖读者先知道前人做法的细节),倾向于按 SPJ
  建议把 Related Work 放在后面,先让读者完整吸收"我的想法",避免过早被一堆背景比较分散注意力
  ——这也更契合 [01 类](01-narrative-structure-and-elevator-pitch.md)讲的"读者优先"原则:先满足
  读者"这篇论文到底讲了什么"的好奇心,再满足"这和其他工作有什么关系"的好奇心
- [ ] 检查会议/期刊的实际惯例——不同子领域、不同审稿人群体的默认预期不同,遵循目标 venue 常见论文的
  惯例,比机械套用某个通用法则更重要
- [ ] 不管放在哪个位置,收尾/定位段落(知识点 4)不能省略——这是位置选择之外唯一不能妥协的部分

**量化验证:** 这条完全是判断力问题,不提供假验证——"这篇论文的 Related Work 该放前面还是放后面"
不存在一个可以用代码判断对错的规则,两种结构在顶会里都大量存在,这里如实标注为纯判断力问题。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"你把 Related Work 放在最后,是真的经过考虑,还是单纯照抄了一个模板?" →
  反驳:如果读者需要读到 Related Work 才能理解 Method 一节里某个设计决策的动机(比如"为什么用这种
  特定的验证器结构"实际上是在回应某篇前人工作的具体局限),却要等到全文快结束才读到这段背景,这就是
  位置选错了的真实信号——判断依据应该是"读者理解方法时是否缺失了背景",不是"哪个位置看起来更规范"。

**常见坑:**
1. 把这条建议当成必须遵守的硬规则,不考虑目标 venue 和具体论文结构的实际需要——七条建议本身也被
   公开质疑过,说明它们是经验法则不是公理。
2. 不管放在哪个位置,都不写知识点 4 讲的收尾定位段落——位置选择解决的是"读者什么时候读到背景信息",
   不能替代"背景信息和自己贡献之间的关系有没有被讲清楚"这件事。

---

## 4. 收尾定位:一句话说清楚"我们相对于这些工作的位置在哪"

不管 Related Work 放在哪个位置,末尾都应该有一段**显式定位**——用近似"The main contributions of our
work relative to X are..."这样的句式,直接回应前文提到的具体局限,而不是让读者自己去拼凑"所以这篇论文
到底比前人多做了什么"。

**真实案例(取材 `research/world-model-imagination-controller/02-deep-gap-analysis.md` §2.4 的
"三者一张表说清楚差距",按设计文档要求做教学化改写,呈现方法而非照搬结论):** 项目组在对比三篇最近邻
工作(AVIC/FFDC/Video-T1)时,没有停留在文字描述"这三篇论文各自做了什么",而是做了一张对照表,把
"该不该生成/该不该采纳/什么时候停/相对不想象基线/相对其他候选/序贯决策/优化目标是否是决策价值"这
几个维度,和"我们的目标"放在同一张表里逐格打勾/画叉——读者一眼就能看出"这三篇论文各自覆盖了哪部分,
唯独'相对其他候选未来的决策价值比较'这一格三篇全部是空的",这正是这篇论文要填的坑。这种"维度对照表"
本质上就是"收尾定位段落"的表格化版本——把散落在几段文字里的比较,压缩成一张可以扫一眼就看懂的表,
对审稿人在"5 分钟看懂核心贡献"这件事上帮助极大(这一点会在 [03 类](03-method-and-results-presentation.md)
进一步展开)。

**可操作检查清单:**
- [ ] Related Work(不管放在前面还是后面)末尾是否有一段显式定位,而不是讲完最后一篇引用就直接结束
- [ ] 定位段落是否**逐条回应**前文提到的具体局限,而不是另起一段泛泛而谈自己的优点
- [ ] 如果对比的前人工作超过 3-4 篇、且比较维度超过 2-3 个,考虑做成表格而不是纯文字——表格能让
  审稿人在几秒内定位到"哪一格是空的",纯文字需要读者自己在脑内构建这张表,认知负担高得多
- [ ] 表格/文字定位是否诚实——不要把前人工作"部分做到"的维度打叉美化成完全没做到,这类夸大一旦被
  审稿人（很可能就是这些相关工作的作者或熟悉者）发现,后果比"承认前人部分做到了"严重得多

**量化验证:** 见知识点 5 的 claims-evidence 检查器——定位段落本身是文字判断,难以直接量化,但"定位
段落提到的每个比较维度,是否在正文其他地方有对应的具体讨论"是可以做交叉检查的,和知识点 5 用的是
同一套方法,这里不重复贴代码,直接引用。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"你的对照表里,FFDC 这一格'相对其他候选'打的是'—'(未覆盖),但 FFDC 的验证器
  其实也隐含了某种候选比较,你确定打'—'不是过度简化?" → 反驳:这正是为什么"逐篇精读原文机制"
  (而不是只读摘要)在做这类对照表之前是必须的——项目组在 `02-deep-gap-analysis.md` 明确说明 FFDC
  "只监控已经生成的那一条想象轨迹",没有"生成多个候选、互相比较、挑一个采纳"这个环节,这个判断建立
  在精读原文机制之上,不是拍脑袋填表;如果审稿人的质疑成立,说明调研深度不够,需要回去重新精读原文
  再下结论,不能靠"表格看起来很专业"蒙混过关。

**常见坑:**
1. 定位段落写成自我表扬("我们的方法显著优于现有工作"),没有具体回应前面提到的局限——这不是定位,
   是空洞的自夸,审稿人不会被说服。
2. 对照表维度选择带有明显的自我服务倾向(专挑自己占优的维度列进表格,回避自己也做不到的维度)——
   有经验的审稿人一眼能看出维度选择是否公允,这类操作反而会降低可信度。

---

## 5. Claims-Evidence 对齐:每条贡献是否在正文里真的有证据支撑

贡献列表写完之后,一个经常被忽略但极其重要的自查步骤是:**逐条检查每条贡献声明,是否在后续的
Method/实验部分有对应的具体证据**。这个检查在论文写作的多个阶段都该做——第一次写贡献列表时做一次
(检验"我打算做的事有没有对应的实验计划"),定稿前再做一次(检验"实验计划有没有变化导致贡献列表
过时")。写作过程本身经常发生这种漂移:实验中途调整方向,但 Introduction 里的贡献列表忘了同步更新,
最后交稿时出现"贡献列表说了 A,正文只字未提 A"的低级失误。

**常见误区/反面例子:** 贡献列表里的第三条声明"我们提供了将本方法与经典 value-of-computation 理论
相连接的理论说明",但正文/附录翻遍都找不到任何和这个理论相关的推导或讨论——这类情况多数不是有意
夸大,而是**写作过程中的真实漂移**:最初计划里有这部分内容,后来因为时间不够被砍掉,但贡献列表忘了
同步删除。

**逐处修改对照:** 检查方式很直接——把贡献列表拆成一条一条的具体主张,逐条问"哪个 Section/Table
/Figure 支撑了这条";找不到对应位置的,要么回去补上对应内容,要么老实从贡献列表里删掉这一条,不能
让贡献列表停留在"曾经计划做但最终没做"的状态。

**可操作检查清单:**
- [ ] 每条贡献是否能标注出对应的 Section/Table/Figure 编号(如知识点 1 改写示例里"(Section 3)"
  这种做法),标注不出来的贡献是漂移的信号
- [ ] 定稿前是否重新核对过一遍贡献列表——写作后期实验/方法经常会调整,贡献列表容易变成"最初计划"
  而不是"最终交付"的快照
- [ ] 反过来检查:正文里有没有分量很重、但贡献列表完全没提到的内容——如果有,说明贡献列表本身
  可能低估了论文真正的贡献,也值得回头调整

**量化验证:** 贡献声明和证据是否真的匹配需要理解语义,不是简单的字符串匹配能完全解决的,但一个
朴素的**关键词重叠检查**可以当作第一道筛子,快速揪出"完全没有任何关键词重叠"的重灾区(重叠为零
几乎可以确定是真的漏了,而不是"写法不同但其实讨论了"这种需要人工确认的边界情况):

```python
import re

def keywords(claim):
    stop = {"we", "our", "the", "a", "an", "of", "to", "that", "than", "and", "is", "in",
            "for", "on", "with", "using", "into"}
    words = re.findall(r"[A-Za-z]+", claim.lower())
    return {w for w in words if w not in stop and len(w) > 3}

def claims_without_evidence(claims, results_text):
    results_words = set(re.findall(r"[A-Za-z]+", results_text.lower()))
    return [c for c in claims if len(keywords(c) & results_words) == 0]

claims = [
    "We propose a controller that adaptively allocates imagination budget.",
    "We show task-conditioned imagination outperforms unconditioned imagination.",
    "We provide a theoretical account connecting our method to value-of-computation theory.",
]

results_text_incomplete = (
    "Table 2 shows our controller's decision-change rate as candidate count K grows. "
    "Table 3 shows the hit rate under task conditioning versus the unconditioned baseline."
)

results_text_complete = results_text_incomplete + (
    " Appendix C derives the Bellman telescoping argument connecting these results to "
    "classical value-of-computation theory."
)

missing_incomplete = claims_without_evidence(claims, results_text_incomplete)
missing_complete = claims_without_evidence(claims, results_text_complete)
assert len(missing_incomplete) == 1
assert len(missing_complete) == 0
print("results_incomplete (bad example, missing evidence for claim 3):", missing_incomplete)
print("results_complete (fixed):", missing_complete)
```

本机实测:`results_text_incomplete` 只覆盖了贡献列表前两条(controller 的决策改变率、task-conditioning
命中率对比),第三条"理论说明"完全没被提到,检查器正确揪出这一条;补上一句提及 Bellman telescoping
推导之后,三条贡献全部能在结果文本里找到关键词重叠,`missing_complete` 为空。**明确边界**:这是关键词
重叠的粗糙代理,不是语义理解——如果正文用完全不同的措辞讨论了同一件事(比如用"decision-value theory"
而不是"value-of-computation theory"),检查器会误报"缺失",最终判断仍然需要人工确认,工具只负责
把"完全零重叠"的高风险位置筛出来供人复核。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你说'我们提供理论说明',这个理论说明具体在论文哪一页?给我一个精确指向。" →
  反驳:如果答不出精确位置,说明这条贡献声明本身站不住,唯一诚实的应对是补充对应内容或者从贡献列表
  删除,没有第三种选择——含糊地说"我们在多处讨论了这个话题"不能算回应,审稿人要的是精确指向。
- **工程约束递增轴(适配成篇幅约束)**:"如果因为页数限制,理论推导被压缩进了附录,贡献列表里还能
  不能理直气壮地写这条?" → 可以,但正文必须有一句明确的指引(如"see Appendix C for the full
  derivation"),不能让读者以为这条贡献完全没有支撑材料——附录内容依然算数,但必须被正文显式引用,
  不能是审稿人自己翻附录目录才偶然发现的。

**常见坑:**
1. 贡献列表在投稿最终版本里,还残留着早期写作阶段计划做、后来因为时间/资源砍掉的条目——这是最常见
   的真实漂移场景,唯一的解法是定稿前强制走一遍逐条核对。
2. 反过来,把正文里权重很轻的一个小实验拔高写进贡献列表凑数——贡献列表的每一条都应该对应论文里
   真正花了篇幅、有分量的部分,不是"技术上确实做了但无关紧要"的边角料。

---

*上一篇:[01-narrative-structure-and-elevator-pitch.md](01-narrative-structure-and-elevator-pitch.md)。
下一篇:[03-method-and-results-presentation.md](03-method-and-results-presentation.md)——Method/实验
部分的呈现逻辑。*
