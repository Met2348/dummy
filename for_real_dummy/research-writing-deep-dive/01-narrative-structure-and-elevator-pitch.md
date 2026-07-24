# 01 · 论文整体叙事结构与电梯演讲(Narrative Structure & Elevator Pitch)

> 总览见 [00-roadmap.md](00-roadmap.md)(六步写作判断力模板说明、五轴追问链的评审对抗改写、真实调研来源、
> 真实项目素材使用边界都在那里,不在每篇正文重复)。本篇是全系列的地基——后面 7 类讲的都是"某一段/某一句
> 该怎么写",这里讲的是"整篇论文的故事线该长什么形状",错了地基,后面每一段写得再工整也救不回来。

---

## 1. C-C-C 框架与 Motivation-Gap-Contribution:同一件事的两种命名

一段好的学术论证,不管是整篇论文、还是一个自然段、甚至一个句子,几乎都能被拆成"背景 → 缺口 → 结论"
这个三段形状。Mensh & Kording(2017,PLOS Computational Biology《Ten Simple Rules for Structuring
Papers》)把它叫 **C-C-C**(Context-Content-Conclusion):Context 交代读者需要的背景、把问题范围收窄到
"这里缺了什么";Content 是你做了什么;Conclusion 收束成"这意味着什么"。国内更常见的说法
Motivation-Gap-Contribution(动机-缺口-贡献)是同一个骨架的另一种命名——Motivation 对应 Context,
Gap 对应"Context 收窄出来的那个缺口",Contribution 对应 Content+Conclusion。两套说法本质上讲的是同一件事,
选哪个命名不重要,重要的是**这三段谁都不能少,且顺序不能乱**——尤其是"缺口"这一段,是很多新手写作里
最容易被跳过的一段:直接从"背景"跳到"我们提出了 X",读者会知道你做了什么,但不知道**为什么值得做**。

**常见误区/反面例子:** 下面这段是一个只有"背景"和"我们做了什么"、完全没有"缺口"的典型反例
(内容围绕"测试时想象预算分配"这个真实问题设定改编,不是逐字照抄任何真实论文):

> World models let an agent simulate future outcomes before acting. Planning with world
> models has been studied extensively. In this paper, we propose a controller that decides
> when to imagine.

三句话读完,读者知道"有个东西叫 world model"、"这个方向有人研究"、"作者提出了个 controller"——但完全
不知道**现有工作到底哪里不够、为什么需要一个新的 controller**。这正是"背景→直接跳贡献"的典型反例,
读者读完的第一反应是"所以呢?"

**逐处修改对照:**

| 原句 | 问题 | 改法 |
|------|------|------|
| "World models let an agent simulate future outcomes before acting." | 背景没问题,但没有收窄到具体缺口 | 保留,作为 Context 第一句 |
| "Planning with world models has been studied extensively." | 空洞背景句,"studied extensively"不提供任何信息增量 | 删除,换成具体指出现有方法的共性做法 |
| "In this paper, we propose a controller that decides when to imagine." | 直接跳到 Contribution,没有 Gap | 插入一句明确的缺口陈述,再引出贡献 |

改写后:

> World models let an agent simulate future outcomes before acting. **Almost every existing
> system spends a fixed computational budget on this simulation, regardless of whether the
> current decision actually benefits from it.** We propose a controller that decides, per
> decision, whether imagining a future is worth its computational cost.

中间插入的一句就是 Gap——不是空洞的"这个领域有很多人做",而是精确指出"大家都用固定预算,不管这次决策
是否真的需要"。有了这一句,最后一句"我们提出了个 controller"才有了存在的理由。

**可操作检查清单:**
- [ ] 全篇/每个大段落是否都能拆出"背景在哪、缺口是什么、结论/贡献是什么"三段,拆不出来说明结构有洞
- [ ] Gap 这一段是否具体到"现有方法共同的做法是 X,这个做法在 Y 场景下会有 Z 后果",而不是"这方面
  研究不够多"这种空话
- [ ] Conclusion/Contribution 是否直接呼应了 Gap 里提出的具体问题,不是另起炉灶讲一件不相关的事
- [ ] 摘要、Introduction 第一段、每个 Section 的开篇段落,分别检查一遍这三层是否都在,C-C-C 是
  **多尺度复用**的结构,不是只在摘要用一次就完事

**量化验证:** "有没有 Gap 信号词、结尾是否收束成结论"本身是判断力问题,但**信号词是否出现**是可以
用正则真实统计的——下面的检查器不判断"这段写得好不好",只检查"C-C-C 三段该有的表层特征是否存在",
是一个粗糙但真实可跑的自查工具:

```python
import re

GAP_SIGNALS = ["however", "but ", "yet ", "unfortunately", "remains unclear", "do not",
               "does not", "fails to", "cannot", "ignore", "no existing", "few works"]
CONCLUSION_SIGNALS = ["therefore", "thus", "as a result", "this means", "we conclude",
                       "this suggests", "consequently", "this shows that"]

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def ccc_shape(paragraph):
    sents = split_sentences(paragraph)
    if len(sents) < 2:
        return {"n_sentences": len(sents), "has_gap_signal": False,
                "has_conclusion_signal": False, "shape_ok": False}
    lower = paragraph.lower()
    has_gap = any(sig in lower for sig in GAP_SIGNALS)
    has_conclusion = any(sig in sents[-1].lower() for sig in CONCLUSION_SIGNALS)
    return {
        "n_sentences": len(sents),
        "has_gap_signal": has_gap,
        "has_conclusion_signal": has_conclusion,
        "shape_ok": has_gap and has_conclusion,
    }

flat_para = (
    "We use a gating network. It takes the current state as input. The gating network "
    "outputs a decision. The decision is either to imagine or not to imagine. We evaluate "
    "this on three tasks."
)

ccc_para = (
    "Adaptive test-time computation lets an agent spend more effort on hard decisions and "
    "less on easy ones. However, existing world-model controllers gate this decision with a "
    "fixed rule learned once at training time and never revisit it per decision, so they "
    "cannot tell a genuinely hard case from an easy one at test time. We therefore train a "
    "gating network that reads the current state and outputs a per-decision imagine/skip "
    "choice, calibrated against a value-of-computation signal. This shows that budget "
    "allocation itself can be made a first-class, per-instance decision rather than a fixed "
    "training-time constant."
)

r1 = ccc_shape(flat_para)
r2 = ccc_shape(ccc_para)
assert r1["shape_ok"] is False
assert r2["shape_ok"] is True
print("flat_para (bad example, flat listing, no C-C-C shape):", r1)
print("ccc_para (revised):", r2)
```

本机实测:`flat_para`(纯罗列句子,五句话平铺,没有转折也没有收束)`has_gap_signal=False`、
`has_conclusion_signal=False`,`shape_ok=False`;`ccc_para`(插入"However..."转折句 + 用"This shows
that..."收尾)两项信号都命中,`shape_ok=True`。**明确边界**:这个检查器只能验证"表层转折词/收束词是否
存在",不能验证"这个转折是否逻辑上站得住"——一段逻辑混乱但恰好用了"however"开头的话照样会被判
`has_gap_signal=True`,这是启发式工具的真实局限,不是可以无视的细节。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:审稿人问"你这个 Gap 到底是'没人做过'还是'有人做过但做得不够好'?这两者审稿人
  的心理预期完全不同"。→ 反驳:必须诚实分清楚——"没人做过"这个说法极其危险,大概率是没查全文献
  (呼应 07 类会讲的"文献调研不足"红旗);更常见也更站得住的 Gap 是"有人做过,但做法有一个具体的、
  可指出的局限",后者才是大多数论文真实的定位方式。
- **规模递增轴**(适配成"这个 Gap 陈述在更大范围的相关工作里还成立吗"):审稿人追问"你说'现有方法都用
  固定预算',真的'都'是吗?举一个反例我就能让这句话站不住"。→ 反驳:Gap 陈述里的全称量词("所有""都")
  是自己给自己埋雷,更安全的写法是精确限定范围("我们核查过的 N 篇 2024-2026 年最相关工作里,没有一篇
  做到 X"),范围越具体,越难被一个反例推翻,也越显得你真的做过系统调研而不是印象流断言。

**常见坑:**
1. 把 Gap 写成"目前的研究还不够多/还不够完善"这种无法证伪的空话——审稿人无法通过读几篇引用文献来验证
   这句话对不对,因为它根本没有具体内容,这种写法几乎不会说服任何一个熟悉该领域的审稿人。
2. C-C-C 只在摘要用一次,Introduction/每个 Section 又退回"背景堆砌→直接讲我做了什么"的平铺写法——
   结构一致性本身就是"这篇论文是精心组织过的"这个信号的一部分,审稿人会下意识注意到这种不一致。

---

## 2. SPJ"先写论文再做研究"的哲学:Don't wait, write

Simon Peyton Jones(微软研究院,Haskell/GHC 的核心设计者)在广为流传的演讲《How to Write a Great
Research Paper》里给出七条建议,第一条就是 **"Don't wait — write"**:传统直觉是 idea → 做研究 → 写论文,
他反过来主张尽早开始写——写作本身会**逼迫你把模糊的想法想清楚**、暴露自己都没意识到的漏洞、并且给了
一个可以拿去和别人讨论的具体对象("opens the way to dialogue with others: reality check, critique,
and collaboration")。他还特别提醒:不要因为"这个 idea 好像不够大"就不敢写——任何 idea,哪怕自己觉得
"weedy and insignificant",都值得先写下来看看。

**常见误区/反面例子:** 最常见的反例不是一段具体的文字,而是一种真实的工作习惯:把"写论文"排在项目
时间线的最后一步——"等实验做完、结果都出来了、方法也定型了,再开始写"。这个习惯的真实代价往往在写作
阶段才会暴露:等到动笔时才发现,论文最核心的那句话("我们到底解决了什么问题")竟然说不清楚——因为这句
话从来没有被逼着写出来过。

**逐处修改对照(这里对照的不是一段文字,而是工作流程的顺序):**

| 传统顺序 | 问题 | SPJ 建议的顺序 |
|---|---|---|
| idea → 做完全部实验 → 才开始写 Introduction | Introduction 里"贡献列表"这个最关键的东西,直到最后一刻
  才被逼着想清楚,如果想不清楚,已经没有时间回头调整实验设计了 | idea 出现后**几天内**就写一版"贡献列表"
  草稿(哪怕实验还没做),用它反过来检验研究计划是否成立 |
| 私下埋头做研究,很晚才拿给同门/导师看 | 别人第一次看到你的思路时,已经投入了大量沉没成本,难以做根本性
  调整 | 写作草稿本身就是"拿去对话"的媒介,越早暴露给别人挑刺,回头调整的成本越低 |

**可操作检查清单:**
- [ ] 有了一个研究想法后,是否在几天内就尝试写出一版"贡献列表"(哪怕只是 3-4 条 bullet,哪怕实验还
  没跑完)——写不出来,往往说明这个想法本身还不够具体
- [ ] 是否把"写"当成研究过程的一部分而不是研究结束后的汇报环节——写作暴露出的逻辑漏洞,是应该反过来
  修正研究计划的信号,不是"写作阶段才要操心的事"
- [ ] 早期草稿是否真的拿给别人看过(哪怕是不熟悉这个子领域的同学),而不是自己反复看到"看起来没问题"
  就停手

**量化验证:** 这一条是**方法论/工作习惯的选择**,不是可以用代码验证对错的陈述——"越早开始写效果越好"
无法在这篇笔记里跑一个 Python 脚本证明,这是如实的判断力问题,不为了凑"可运行例子"硬造一个假 assert。
唯一能诚实做的量化,是给出一个可执行的自检动作:如果一个想法**写不出一句可证伪的贡献陈述**(见下一个
知识点"只讲一个核心贡献"里对"可证伪"的定义),这本身就是一个客观可检查的信号,提示这个想法可能还没
想清楚——这个检查动作会在知识点 4 用代码演示,这里不重复。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"这不就是本末倒置吗?研究都没做完,怎么知道要写什么?" → 反驳:SPJ 的建议不是
  "研究没做就编结果",而是"结论性的贡献陈述"和"支撑它的具体实验"是两件可以交替推进的事——先写一版
  贡献陈述,如果发现某条贡献目前的实验设计完全支撑不了,这恰恰是在实验阶段还有调整空间时就发现问题,
  比写到 Introduction 最终稿时才发现要便宜得多。
- **真实性验证轴**:"你举的这个'早期草稿倒逼研究设计'的说法,有没有真实案例,不是听起来有道理就是有道理"
  → 反驳:世界模型想象预算控制器这个真实项目里就有一个案例(`02-deep-gap-analysis.md`)——项目组把
  pilot 阶段的发现写成"会前简报"文档的过程中,倒逼自己重新审视"发现一/二"到底新不新,结果发现它们是
  1991 年 Value of Computation 理论的直接推论,这个认识不是先做完全部研究才想到的,而是在**写文档、
  组织论证**的过程中被逼出来的——这正是"写作暴露漏洞"这条原则的真实体现,05/06/07 类会从不同角度
  再次引用这同一个真实案例。

**常见坑:**
1. 把"早写"理解成"早交/早发表"——SPJ 讲的是尽早开始写**草稿**用于自我检验和寻求反馈,不是缩短完整的
   研究周期或跳过严谨性,这是两件事,不要混淆。
2. 只写了一次就再也不回头改——"写作暴露问题"这件事只有在写完后真的认真重新读、认真找人提意见的前提下
   才成立,写完锁进抽屉不会自动产生任何效果。

---

## 3. 摘要电梯演讲公式:Context → Gap → Contribution → 量化结果

摘要是全篇读者最多的部分(很多人只读摘要就决定要不要继续读),结构应该高度稳定,不应该每次重新发明。
一个能实际操作的公式(综合多个来源的共识,而不是单一来源):**一句话交代背景/大方向 → 一句话点出现有
方法共同的缺口 → 一到两句话讲清楚你的方法 → 一句给出关键量化结果**,总长度落在 150-300 词区间(6-7
句话上下),不需要引用文献(摘要里出现"[3]"这种引用标记是常见反模式),不能引入正文没有的新论点。

**常见误区/反面例子:** 下面是一段典型的"看起来像摘要,实际信息量很低"的反例:

> World models are an important part of modern reinforcement learning and have been studied
> extensively in recent years. Imagination-based planning is a promising direction for
> improving decision making. In this paper, we study the problem of imagination in world
> models. We propose a new method and conduct experiments to evaluate its performance. Our
> approach is general and can be applied to many settings. Experimental results demonstrate
> the effectiveness of our method.

这段话把摘要该出现的"背景/缺口/方法/结果"四件事全部替换成了通用套话——"an important part"、"studied
extensively"、"a new method"、"the effectiveness of our method"——读完之后完全不知道这篇论文和同方向
其他论文有什么区别,任何一篇 world model 论文的摘要把方法名换掉都能通用这段话,这是"泛泛而谈"最典型
的样子。

**逐处修改对照:**

| 原句 | 问题 | 改法 |
|---|---|---|
| "World models are an important part of... studied extensively." | 通用背景,不收窄到具体问题 | 换成
  一句精确描述"world model 具体在做什么决策" |
| "In this paper, we study the problem of imagination." | 没有指出缺口,"study the problem"不是缺口 | 
  插入一句具体的 Gap:现有方法共享的做法+这个做法的后果 |
| "We propose a new method and conduct experiments." | "a new method"没有信息量 | 具体说清楚方法的
  关键机制是什么 |
| "Experimental results demonstrate the effectiveness." | 没有任何数字,无法被记住也无法被核查 | 换成
  具体的量化对比数字 |

**可操作检查清单:**
- [ ] 通读一遍,划掉所有"an important part"/"has been studied extensively"/"the effectiveness of
  our method"这类可以套用到任何一篇同方向论文的句子——凡是划掉后论文的辨识度没有变化的句子,都是
  可以删除或者替换成具体内容的信号
  (呼应 [approximatelycorrect.com 的 ML 写作建议](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/):"审稿人一年读 50-100 篇同方向论文,通用开场白只会让他们觉得无聊")
- [ ] 是否有至少一处具体数字(百分比、倍数、样本量),没有数字的摘要很难被审稿人记住,也没法被读者
  快速判断"这个提升值不值得看正文"
- [ ] 摘要是否可以脱离正文独立成立——不能包含"如图 3 所示"这种依赖上下文的表述,也不能包含引用标记
- [ ] 摘要和 Introduction 第一段是不是几乎同一段话——两者应该服务不同目的(摘要是独立的浓缩总结,
  Introduction 是带着读者一步步进入问题),完全重复说明至少有一处没有发挥该有的作用

**量化验证:** "这段摘要吸不吸引人"是判断力问题,但"结构该有的表层特征是否齐全"可以用代码检查——
下面的检查器验证四件事:词数是否落在常见区间、是否包含 Gap 转折信号、是否包含具体的量化结果、贡献句
(以 We/Our 开头)数量是否合理(0 条说明读者抓不住重点,过多条则可能是"塞了太多 idea 没有取舍",呼应
知识点 4):

```python
import re

GAP_SIGNALS = ["however", "but ", "yet ", "unfortunately", "remains unclear", "do not",
               "does not", "fails to", "cannot", "regardless of whether", "ignore"]
RESULT_SIGNALS = [r"\d+(\.\d+)?\s?%", r"\d+(\.\d+)?[x×]\b", r"\d+(\.\d+)?\s?(pp|percentage points)"]

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def check_abstract(text):
    words = re.findall(r"[A-Za-z']+", text)
    sents = split_sentences(text)
    lower = text.lower()
    has_gap = any(sig in lower for sig in GAP_SIGNALS)
    has_result = any(re.search(pat, text) for pat in RESULT_SIGNALS)
    contribution_sents = [s for s in sents if re.match(r"^(We|Our|This paper|This work)\b", s)]
    return {
        "word_count": len(words),
        "n_sentences": len(sents),
        "in_150_300_band": 150 <= len(words) <= 300,
        "has_gap_signal": has_gap,
        "has_quantitative_result": has_result,
        "n_contribution_sentences": len(contribution_sents),
    }

vague_abstract = (
    "World models are an important part of modern reinforcement learning and have been "
    "studied extensively in recent years. Imagination-based planning is a promising "
    "direction for improving decision making. In this paper, we study the problem of "
    "imagination in world models. We propose a new method and conduct experiments to "
    "evaluate its performance. Our approach is general and can be applied to many "
    "settings. Experimental results demonstrate the effectiveness of our method."
)

structured_abstract = (
    "World models let an agent search over imagined futures before it acts, but almost "
    "every existing system spends a fixed computation budget on this search regardless of "
    "whether the current decision benefits from it. We show that when the imagined rollout "
    "and the agent's baseline policy share exactly the same model, spending more computation "
    "at decision time does not change the expected outcome and only adds variance: across five "
    "seeds, the decision-change rate falls from 35.6% to 10.0% as the number of sampled "
    "candidates grows from 1 to 10, while over 65% of changed decisions make things worse. "
    "We then show that giving the rollout a genuine information advantage over the baseline "
    "flips this result: task-conditioned imagination reaches an 82.0% hit rate versus 63.7% "
    "for unconditioned imagination across three target settings. Building on this, we propose "
    "a controller that allocates imagination budget based on estimated information advantage "
    "rather than a fixed schedule, and we outline how this connects to classical value-of-"
    "computation theory."
)

r_vague = check_abstract(vague_abstract)
r_structured = check_abstract(structured_abstract)
assert r_vague["has_gap_signal"] is False and r_vague["has_quantitative_result"] is False
assert r_structured["has_gap_signal"] is True and r_structured["has_quantitative_result"] is True
assert r_structured["in_150_300_band"] is True
print("vague_abstract (bad example):", r_vague)
print("structured_abstract (revised):", r_structured)
```

本机实测:`vague_abstract` 只有 72 词(明显偏短,信息密度低到没什么可展开的),`has_gap_signal=False`、
`has_quantitative_result=False`;`structured_abstract` 159 词、落在 150-300 区间内,`has_gap_signal=True`
(命中 "regardless of whether")、`has_quantitative_result=True`(命中 "35.6%"/"63.7%" 等)。
`structured_abstract` 里的 82.0%/63.7% 是 world-model 项目 `01-meeting-briefing.md` 里
task-conditioning pilot 的真实结果数字,这里按设计文档的边界使用规则直接引用真实数字支撑"这是真实科研
判断力长什么样",但摘要文本本身是围绕这个真实场景**重新撰写**的教学示例,不是从项目文件里逐句复制。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"你的摘要塞了两个发现(H 越深命中率不升反降 + task-conditioning 能反超),到底
  哪个才是这篇论文的核心贡献?" → 反驳(呼应知识点 4):如果确实有两个独立价值的发现,需要在摘要里
  明确谁是主线、谁是支撑性证据,不能让读者自己猜;如果两者权重相当到没法取舍,这本身就是"该不该拆成
  两篇论文"的信号,不是摘要写作技巧能解决的问题。
- **真实性验证轴**:"82.0% vs 63.7% 这两个数字,置信区间/种子数在哪?摘要里只放点估计,正文能不能
  兜住?" → 反驳:摘要受长度限制确实通常只放点估计,但正文和图表必须能兜住这个数字的方差声明(呼应
  07 类会讲的"无种子方差"红旗)——摘要给出的数字必须是正文里能追溯到具体表格/图的数字,不能是摘要
  单独拔高的版本。

**常见坑:**
1. 摘要里出现引用标记(如 "[3]")——摘要应该独立于参考文献列表被理解,大部分期刊/会议格式规范也明确
   不建议摘要出现引用。
2. 摘要和 Introduction 大段重复——两者应该是"独立的浓缩总结"和"带读者一步步进入问题"两个不同功能,
   逐句雷同说明至少有一处没有发挥该有的作用。
3. 摘要引入了正文完全没有展开的新论点或新数字——审稿人一旦发现"摘要说的这件事正文根本没有对应支撑",
   会立刻怀疑整篇论文的严谨性,这个信任一旦丢失很难在后续段落挽回。

---

## 4. 只讲一个核心贡献:SPJ 的"one key idea"与真实的叙事重新定位案例

SPJ 七条建议里的"nail your contributions"背后有一个更尖锐的说法:**一篇论文最终只能让读者记住一件事**
("you only get to say one thing")。如果手上有四个值得讲的 idea,更诚实的做法是写四篇论文,而不是把
它们全部塞进一篇、指望读者自己去拼凑出"到底哪个才是重点"。贡献陈述还必须**可证伪**(refutable)——像
"We describe the WizWoz system. It is really cool"这种说法审稿人无法判断对错,而"我们的方法在 X 指标上
比最强基线高 Y 个百分点"是一句读者读完就能去验证的具体主张。

**常见误区/反面例子:** 最常见的反模式不是"贡献列表写得不好",而是**贡献本身就没收敛**——手上有 3
个互相独立的发现,论文摘要/Introduction 试图给三者相同的权重,结果读者读完记不住任何一个,因为大脑
没有办法同时记住三件互不relate的"重点"。

**逐处修改对照(这是一个真实发生过的叙事重新定位案例,取材 `research/world-model-imagination-controller/02-deep-gap-analysis.md`,细节按设计文档要求做了教学化改写,不逐字复制原文结论):**

项目组在一次真实的诊断性 pilot 里,同时得到了三类观察:①想象和基线完全同源时,想象深度越深、决策
反而越容易被带偏("发现一");②给想象加入和决策无关的随机性没用("发现二");③给想象一个真正的、
决策相关的信息优势后,想象命中率能稳定反超基线("发现三")。**最初的会前简报草稿把叙事重心放在
"发现一/二"上**——这两条读起来更"反直觉"、更适合当开篇的钩子。但项目组在后续更深入的文献核查阶段
发现:发现一/二的核心原理,正是 Russell & Wefald(1991)Value of Computation 理论的经典停止法则,
已经被一篇 1991 年之后的 UAI 2012 论文形式化过——**而这篇论文本来就在自己的文献库里**。如果继续把
"发现一/二"当核心贡献去讲,审稿人只要熟悉这段理论,几乎必然会指出"这是 reinventing the wheel",这是
典型的 desk reject 成因。

| 版本 | 核心贡献选择 | 问题 |
|---|---|---|
| 最初草稿 | 发现一/二("想象和基线同源时,多算不会变好") | 读起来反直觉、抓眼球,但核心原理是 1991 年
  经典理论的直接推论,不是新发现——一旦被审稿人识别出来,"新颖性"这条主张会直接站不住 |
| 重新定位后 | 发现三("给想象真正的信息优势,命中率能稳定反超,且信息优势的渗透是连续的、不是开关式的") | 
  穷尽检索没有找到先例,是三条发现里唯一站得住"系统性新发现"这个说法的一条 |

**关键点不是"发现一/二就没有价值了"——而是价值的性质变了**:项目组把它们重新定位成"理论基线与阴性
对照"(第一次在这个具体场景下用严格数学把一个已知原理钉实、做出可复现实验),这本身是合法贡献,只是
不能再当"新发现"来卖。**诚实地讲清楚这个定位,反而是在向审稿人证明做过认真的文献调研,是加分项,不
讲清楚才是减分项**——这条判断直接呼应 05 类会展开的"局限性自曝"和 07 类会展开的"reinventing the
wheel 红旗"。

**可操作检查清单:**
- [ ] 如果贡献列表有 3 条以上,能不能用一句话讲清楚"这些贡献共同服务于同一个中心论点是什么"——讲不出
  这句话,说明贡献之间可能是并列而非递进,需要考虑拆分或者重新排序权重
  (来源:SPJ 演讲评论区总结的核心主张,"you only get to say one thing")
- [ ] 每条贡献是否可证伪——能不能写成"我们在 X 场景下测得 Y",而不是"我们展示了一个很酷的系统"这种
  无法验证对错的说法
- [ ] 有没有诚实核查过:自己认为的"新发现",是不是已经是某个更早理论的推论——核查方式是回头翻自己
  已经收集的文献库(项目组这次就是在自己的文献库里发现了 1991 年理论的正式化版本),不是假设"我没
  读到过就等于不存在"
- [ ] 如果核查后发现某条贡献不再是"新发现",是否愿意重新定位它的价值(比如变成"阴性对照"/"理论基线"),
  而不是硬着头皮继续包装成新颖性主张

**量化验证:** "哪个发现才是核心贡献"是判断力问题,不可 assert;但"贡献列表条数是否收敛到一个可管理
的范围、每条是否包含可证伪的具体数字"是可以检查的表层信号,复用知识点 3 检查器里的
`n_contribution_sentences` 字段即可(不重复贴代码)——如果一段摘要里以 "We/Our" 开头的贡献句超过
4-5 条,大概率是没有做"只讲一个核心贡献"这个取舍,值得回头检查是不是该拆分或者调整权重。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你说这是'系统性新发现',查过的文献库有多大?怎么证明不是你没读到相关工作?" →
  反驳:诚实的答案只能是"穷尽检索没有找到先例"而不是"确认不存在"(呼应
  `04-sharpened-recommendation.md` 里项目组自己的措辞:"这是一个'未找到'而非'确认不存在'的判断")——
  过度自信地宣称"世界上没人做过"是审稿人最容易当场证伪的一类说法,承认调研的边界反而更可信。
- **决策依据追问轴**:"你把发现一/二从'核心贡献'降级成'理论基线与阴性对照',这是不是纯粹的话术包装,
  内容完全没变?" → 反驳:不是话术包装,是评价标准变了——当"新发现"来卖,评价标准是"这个结论此前
  没人证明过";当"理论基线与阴性对照"来卖,评价标准变成"是否在一个新的具体场景下用严格方法把已知
  原理做实、是否可复现"——项目组用鞅的全方差分解重新推导 Bellman telescoping、并在受控环境里出了
  可复现实验,这满足的是第二套标准,不是回避第一套标准不达标的事实。

**常见坑:**
1. 舍不得放弃某个"听起来很酷但撑不住新颖性审查"的贡献,试图靠模糊的措辞蒙混过关——审稿人里总有人
   足够熟悉相关理论,一旦被当场指出,论文的可信度会连带其余真正站得住的贡献一起受损。
2. 贡献列表条数堆到 6-7 条,试图靠"数量多显得工作量大"来加分——效果通常相反,读者记不住任何一条,
   反而显得没有取舍能力。

---

## 5. 标题写作判断力:具体 vs 花哨

标题是比摘要还要早被读到的一层——很多检索场景下(学术搜索引擎、社交媒体转发)读者只会看到标题就决定
点不点进来。好标题的判断力核心是:**精确地讲清楚"是什么 + 新在哪",而不是靠修辞制造悬念或者堆砌
时髦词**。

**常见误区/反面例子:** 两类常见反例:①过度宽泛,读完不知道具体做了什么("A Study on World Models
for Decision Making");②为了追求"抓眼球"堆砌流行词而牺牲精确性("Imagination Is All You Need:
Rethinking World Models via Adaptive Meta-Cognitive Test-Time Scaling")——后者看起来信息量很大,
实际上"Meta-Cognitive"这类词在标题里没有被论文内容严格对应定义,容易被审稿人视为夸大。

**逐处修改对照:**

| 反例 | 问题 | 改法 |
|---|---|---|
| "A Study on World Models for Decision Making" | 任何一篇 world model 论文都能用这个标题,没有辨识度 | 
  加入具体机制词:"Adaptive Imagination Budget Allocation for World-Model Planning" |
| "Imagination Is All You Need: Rethinking World Models via Adaptive Meta-Cognitive
  Test-Time Scaling" | 堆砌流行语("...Is All You Need"这个句式本身已经被过度使用)、"Meta-Cognitive"
  这个词比论文实际机制更宏大 | 去掉修辞外壳,标题只保留论文里真正定义过的机制名词 |

**可操作检查清单:**
- [ ] 标题里出现的每一个关键词,是否都能在论文正文里找到对应的、被严格定义过的内容——标题不能比
  论文本身"讲得更大"
- [ ] 去掉标题里的修辞句式("X Is All You Need"、"Rethinking X"、"Towards X"这类句式已被大量使用,
  除非确实贴切,否则容易显得跟风而不是精确)
- [ ] 标题能否让一个同方向但不是这个子问题的研究者,在 5 秒内判断"这篇论文和我的方向有没有关系"

**量化验证:** 标题好坏本身是判断力问题(呼应 00 类roadmap里说明的边界),这里如实不提供一个假验证——
"标题吸不吸引人"和"标题是否夸大"都需要读者对该子领域有判断力才能评估,没有一个正则表达式能可靠地
区分"恰当的精炼"和"华而不实的夸大",强行写一个"检测浮夸词列表"的脚本只会制造虚假的确定性(比如
"Meta-Cognitive"这个词本身不是禁用词,如果论文真的严格定义并使用了这个概念,用在标题里就是恰当的)。
唯一站得住的量化,是复用知识点 3 的检查逻辑做一次**长度检查**——多数会议标题建议控制在 15 个词以内,
过长的标题(尤其是副标题堆砌多个冒号分句)本身是一个可以客观计数的信号,虽然它只是一个弱代理,不是
判断标题好坏的充分条件。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"你标题里用了'Adaptive',论文里所有配置都需要手动调超参数,这个'Adaptive'名副
  其实吗?" → 反驳:如果审稿人的质疑成立(方法本质上仍需要人工调参,只是调参空间比 baseline 更大),
  这是标题夸大的真实红旗,需要老实改成更保守的说法(比如"Budget-Aware"而非"Adaptive"),不能靠"大家
  都这么用这个词"来辩护。
- **真实性验证轴**:"标题里的关键词,在正文摘要之外的地方还出现过几次?" → 这是一个可以真实统计的
  检查(词频统计),如果标题关键词在正文里只出现在摘要,再没在 Method/实验部分被具体定义或使用过,
  说明标题这个词很可能是"贴上去"的包装,而不是论文真正的核心机制。

**常见坑:**
1. 标题和摘要的第一句用完全不同的词描述同一件事(比如标题说"gating",摘要说"routing")——不是错误,
   但会增加读者的认知负担,统一术语能让标题、摘要、Introduction 三层的呼应更强。
2. 为了"听起来更厉害"在标题里使用还没有被论文充分证明的宏大概念词(如"General"/"Universal"),而
   实验只覆盖了 1-2 个受控场景——这类词一旦被审稿人追问"泛化到了哪些设置",很容易反过来伤害论文
   的可信度。

---

*下一篇:[02-introduction-and-related-work.md](02-introduction-and-related-work.md)——Introduction
段落级写作与 Related Work 定位。*
