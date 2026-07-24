# 06 · 会议日 Capstone:从摘要被接收,到 Q&A 被问到犀利问题当场怎么接

> **这是一个假设性场景,不是真实已经发生的会议经历。** 用户真实在研项目
> `research/world-model-imagination-controller/`(世界模型测试时想象预算自适应分配控制器)截至目前
> 还处于投稿准备阶段,尚未在任何会议上做过报告——这篇 capstone 借用该项目文档里已经真实存在的论证
> 结构、真实数字、真实发生过的"自我核验"过程(比如
> [`02-deep-gap-analysis.md`](../../research/world-model-imagination-controller/02-deep-gap-analysis.md)
> 里记录的、项目组自己发现"某个'新发现'其实是 30 年前经典理论的推论"这件真实的事),设计成一个完整
> 的"假设这篇论文被接收之后,报告人会经历什么"的教学场景,演示本系列 01-05 号文件讲的方法论怎么
> 串起来真正用一遍。这不是代写用户真实论文的内容,也不是编造一个"某次会议上真的发生过"的假事件。

**串联的分类**:01(spotlight 时间预算)→02(slide 信息密度)→03(poster 版面+电梯讲解)→04(Q&A 应对,
含 CAP 结构与承认合理批评)→05(networking 与会后跟进)。

---

## 背景:摘要被接收

**假设的场景设定**:项目组把"idea 10"(诊断性研究——想象到底什么时候真的有用)这条线整理成一篇独立
的短文投出去。选这条线做 capstone 的场景,不是随便选的——项目自己的调研文档里明确记录过,这是 10 个
候选方向里"几乎必做"、"风险最低"的一条(见
[`01-meeting-briefing.md`](../../research/world-model-imagination-controller/01-meeting-briefing.md)
§4 的候选表),很适合作为一篇诚实、范围声明清楚的早期论文独立发出去,不需要等主线方法论文一起投。

投稿结果:**poster + 5 分钟 spotlight talk**——一种常见的"两段式"接收方式,给了报告人一个上台露脸
的机会,但时长很短,论证节奏必须极度精简。

**这篇论文的核心内容(取材真实项目材料,细节简化改编)**:
- 核心结论:想象是否有用,取决于它有没有携带基线不知道的、真正和任务相关的信息——单纯多算不会变好
  (这条背后有完整的 Bellman telescoping 数学推导支撑);给想象一个真正的信息优势(比如让它知道当前
  真实任务目标),想象命中率能从 63.7%(unconditioned)稳定提升到 82.0%(task-conditioned)。
- 诚实的范围声明:这是一个 32 状态、5 个随机种子的表格化合成环境上的原型 pilot,不是最终大规模实验。
- 已经完成的自我核验:项目组自己发现,pilot 里"想象和基线共享同一个不完美模型时,多算不会变好"这条
  结论,本质上是 Russell & Wefald 1991 年 Value of Computation(VOC)理论的直接推论,不是全新发现。

---

## 第一阶段:准备 5 分钟 Spotlight Slides(应用 01、02 号文件方法论)

5 分钟是一个极端苛刻的时长——按 01 号文件的语速公式(140 词/分钟),预算只有约 700 词。第一版草稿
几乎必然超支,这是真实、普遍会发生的第一次尝试,不是准备不认真:

```python
def estimate_minutes(word_count, wpm=140):
    return word_count / wpm

draft_before = {
    "motivation": 120,
    "related_work_recap": 300,
    "eval_protocol_detail": 330,
    "one_big_finding": 200,
    "limitations": 100,
}
draft_after = {
    "motivation": 90,
    "one_big_finding_with_example": 420,
    "scope_caveat": 100,
    "closing": 90,
}

SPOTLIGHT_BUDGET_MIN = 5
WPM = 140

def audit(draft, label):
    total_words = sum(draft.values())
    minutes = estimate_minutes(total_words, WPM)
    idea_words = sum(v for k, v in draft.items() if "finding" in k)
    idea_share = idea_words / total_words
    print(f"{label}: total_words={total_words} est_minutes={minutes:.2f} idea_share={idea_share:.2f}")
    return minutes, idea_share

before_min, before_share = audit(draft_before, "before (5-min spotlight draft)")
after_min, after_share = audit(draft_after, "after")

assert before_min > SPOTLIGHT_BUDGET_MIN + 2, "the before draft should clearly overshoot the 5-minute budget"
assert before_share < 0.3, "the before draft's core-finding share should be clearly too low"
assert after_min <= SPOTLIGHT_BUDGET_MIN + 1, "the after draft should basically fit the 5-minute budget"
assert after_share >= 0.5, "the after draft should give the core finding most of the time"
print("ALL SPOTLIGHT-BUDGET ASSERTIONS PASSED")
```

**改前砍掉了什么、为什么**:`related_work_recap`(和 AVIC/FFDC/Video-T1 逐篇比较)整段砍掉,按
01 号文件第 1 节的原则,压缩成 Q&A 里才展开的储备内容;`eval_protocol_detail`(合成环境的具体设计
细节)只留一句话带过,详细内容留给论文正文。剩下的时间几乎全部给了"一个大发现 + 一个例子",这正是
01 号文件反复强调的"talk 只讲一个 idea"。

**Slide 设计**(应用 02 号文件):最终定稿的核心 slide,就是本系列 02 号文件第 1 节展示过的那张
`good_slide.png`——一句话结论("Imagination only helps if it knows something the baseline doesn't")
+ 一个大号数字("82.0% vs 63.7%")。这不是巧合式的重复,是刻意的设计延续:capstone 用的就是 02 号
文件已经真实渲染、量化验证过的那张 slide 作为这篇假设论文的真实 slide 内容。

---

## 第二阶段:准备 Poster(应用 03 号文件方法论)

Poster 环节同样直接复用本系列已经产出的真实素材——03 号文件第 1 节的 `good_poster.png`,标题是
一整句结论("Imagination only pays off when it carries task-relevant information the baseline
lacks."),中央大留白区放核心数字,两侧窄栏放 Method / Related Work / Results Detail / Contact 四块
细节。电梯讲解按 03 号文件第 3 节的 ABT 结构准备了 30 秒版本:

> "世界模型的想象预算,现在几乎都是写死的常数——**但**我们发现,想象只有在真的比基线多知道点什么的
> 时候才划算——**所以**我们做了一套诊断性实验,把'想象什么时候真的有用'从直觉变成了可以测量的数字。"

---

## 第三阶段:Spotlight 当天——5 分钟讲完,进入 Q&A

报告按准备好的节奏讲完,压线在 5 分钟内收尾,回到开场那句话("如果想象不比基线多知道点什么,多算就是
在做无用功")。主持人开放提问,连续 4 轮追问接踵而至——这不是运气不好,一篇诚实做了范围声明的早期
论文,天然会招来这类问题,这正是准备阶段就该预判到的。

### 追问 1(决策依据追问轴):"这不就是经典的 Value of Computation 理论吗?"

**听众**:"你说'想象要有信息优势才划算'——这不就是 Russell and Wefald 1991 年 Value of Computation
理论的直接推论吗?你们的新意到底在哪?"

**报告人**:"这个问题问得很准——我们自己在准备阶段也查证过这一点,确认这条结论本质上确实是经典 VOC
理论的一个推论,不是全新发现。我们做的是第一次把这个 30 年前的原理,在'测试时想象+现代世界模型'这个
具体场景里用严格数学钉实、做出可复现的受控实验——这是一个诚实的理论基线贡献。真正的新意在另一条
结论上:我们发现给想象的信息优势不是开关式的,是连续的、分通道的——这条我们检索之后没有找到先例。"

**小结**:这正是 04 号文件第 3 节讲过的"承认+重新定位"结构——先正面承认问题指出的事实是对的,不
含糊、不硬辩,再清楚讲明白这部分工作真实的定位,把话题引导到真正站得住的新意上。这个回答之所以能
从容给出,前提是准备阶段(第一阶段之前)就已经做过这次自我核验,不是被问到才第一次意识到这一点。

### 追问 2(真实性验证轴):"32 个状态、5 个种子,这个结论能信吗?"

**听众**:"你们的 pilot 只在一个 32 状态的合成环境里测过,5 个种子——这个规模的结论,能推广到真实
场景吗?"

**报告人**:"不能直接推广,这一点我们在论文里写得很明确。选合成环境是因为需要一个可以精确算出
ground-truth 最优解的'裁判',这样才能干净地隔离'想象有没有用'这一个问题,不会和'评测本身准不准'
混在一起——这是方法论上的取舍,不是偷懒。规模小是真实的限制,我们没有回避它:这是一个原型规模的
pilot,不是最终结论,下一步就是把同样的诊断方法搬到真实 world model(比如 TD-MPC2)上重新验证。"

**小结**:这条回应直接来自项目真实文档里已经写好的"范围声明"原话精神(参见
[`01-meeting-briefing.md`](../../research/world-model-imagination-controller/01-meeting-briefing.md)
§3.6:"这是原型规模的 pilot(32 状态、5 种子、表格模型),不是论文最终实验")——诚实的范围声明不是
临场发明的话术,是写论文/准备 slide 阶段就该想清楚、写清楚的内容,Q&A 现场只是把已经想清楚的东西
再说一遍,而不是第一次面对这个问题。

### 追问 3(方案批判迭代轴):"你们真的跑过和 FFDC/AVIC 的对比实验吗?"

**听众**:"你们提到 AVIC、FFDC 这些最近邻工作,但今天讲的结果里没看到直接对比数字——你们真的跑过
这些 baseline 吗?"

**报告人**:"目前没有跑完整对比,如实说明:我们做过一轮基线可复现性调研,发现这几篇工作的可复现
门槛差别很大——有两篇(Video-T1、Finding the Time to Think)代码、权重、benchmark 全部公开,可以
直接跑;AVIC 的权重需要额外审批加大显存;FFDC 关键的 verifier 超参数论文没有披露,复现难度最高。
我们目前这篇是诊断性研究,核心目的是测量'想象什么时候有用'这个问题本身,不是发起一次新的门控方法
和这几篇打榜——真正对比是我们下一步方法论文要做的事,这次这篇论文里没有回避说'我们对比过了',而是
清楚写明这是留给后续工作的。"

**小结**:这条回应的底气来自项目真实完成过的一次"诚实盘点"——
[`07-baseline-reproducibility-audit.md`](../../research/world-model-imagination-controller/07-baseline-reproducibility-audit.md)
逐篇记录了每个 baseline 到底能不能跑、卡在哪一步,这类"我们调研过复现难度,如实告诉你现状"的回答,
比含糊地说"我们对比过了"或者被问到才慌乱承认"没有对比"更专业——它展示的是"知道自己工作边界在
哪",这本身是加分项。

### 追问 4(带压力的追问,04 号文件第 4 节场景):"既然不是新发现,这个工作还有什么发表价值?"

**听众**(语气略带挑战):"你自己都承认了,这条核心结论不是新发现——那这个工作凭什么该被接收?"

**报告人**:"如果一个理论结论只有'是不是全新的'这一个评判标准,那确实很多严谨的实证/基线类工作
都不该存在。这篇论文的价值不在提出一个新原理,在于**第一次把一个已知原理,在这个具体、此前没人
系统测量过的场景里,做成可复现的受控实验、给出具体数字**——这本身是很多后续工作(包括我们自己的
下一步方法论文)会依赖的地基。而且我们并没有止步于复现已知结论:報告里提到的第二条发现(信息优势
的渗透是连续、分通道的),是我们检索之后没有找到先例的部分,这才是我们主张的新意所在。"

**小结**:这一轮延续追问 1 已经承认过的局限,但被追问得更直接、更有压力——正是 04 号文件第 4 节
讲的"专业冷静地把攻击性提问转成具体技术问题"的应用场景:回应没有因为对方语气更强硬就改变实质内容
(该承认的部分依然大方承认),也没有被牵着走向情绪化对抗,而是再次清楚地把"诚实的基线贡献"和
"真正的新意"分开陈述。

### 如果提前做过 04 号文件第 1 节的"三档分类法"准备,会是什么样

```python
def bucket_questions(questions):
    buckets = {"confident": [], "difficult": [], "dreaded": []}
    for q, confidence in questions:
        if confidence >= 0.8:
            buckets["confident"].append(q)
        elif confidence >= 0.4:
            buckets["difficult"].append(q)
        else:
            buckets["dreaded"].append(q)
    return buckets

qa_prep = [
    ("Isn't this just the classic Value of Computation theory -- where's the novelty?", 0.65),
    ("Your pilot only has 32 states and 5 seeds -- can this conclusion be trusted?", 0.5),
    ("Compared to FFDC/AVIC, have you actually run the comparison experiments?", 0.3),
    ("Given you admit this isn't a new finding, what's the publication value here?", 0.35),
]

buckets = bucket_questions(qa_prep)
for k, v in buckets.items():
    print(f"{k}: {v}")

assert len(buckets["confident"]) == 0, "none of these should be 'definitely can answer' -- honestly reflects this is a tough round"
assert len(buckets["difficult"]) == 2
assert len(buckets["dreaded"]) == 2
print("ALL CAPSTONE-QUESTION-BUCKET ASSERTIONS PASSED")
```

**如实说明**:这场 Q&A 里没有一条问题落在"肯定答得上来"这一档——这不是设计失误,是诚实反映了
"诊断性研究+主动做过自我核验"这类论文天然会招来的追问强度:承认过局限的工作,现场被追问得更细
是正常代价,不是准备不够的信号。真正决定这几轮回应能不能站住的,不是"有没有被问到难题",是
"准备阶段有没有已经想清楚这些问题的答案"——四轮追问的回应内容,没有一句是报告人临场编出来的,
全部来自准备阶段(甚至更早的项目真实文档)就已经想清楚、写清楚的内容。

---

## 第四阶段:Poster Session 与会后跟进(应用 05 号文件方法论)

Spotlight 结束后是下午的 poster session。报告人的海报旁边围了几位听众,其中一位是做 MuZero 系统
方向的博士后,提到自己一直想知道"模拟预算能不能也用类似的诊断方法测一遍"——这是一次真实、具体的
对话,不是"加个联系方式"式的打卡社交(呼应 05 号文件第 2 节)。报告人当场没有现成答案,但记下了
对方的方向,约定"这个问题我们目前没测过,回去想一下要不要作为下一步的一个分支,再联系你"。

会议结束后三天内,报告人发了一封简短、具体的后续邮件,提到当天聊过的"MuZero 模拟预算能否用同样
诊断方法测量"这个具体想法(不是"很高兴认识你"这类模板客套话),约定关注对方后续是否有相关进展——
这是 05 号文件第 4 节强调的"具体细节证明你真的记得这次交流"的直接应用。

---

## 复盘:如果重来一次

把整场经历倒回去看,几个真正决定这一天顺不顺利的因素,全部发生在报告人真正走上台之前:

1. **准备 slide 时提前做的时间预算审计**(01 号文件),让 5 分钟的极限时长没有变成"讲到一半被打断"
   的事故。
2. **准备阶段主动做过的自我核验**(项目真实文档 `02-deep-gap-analysis.md` 记录的过程),让追问 1
   和追问 4 这两轮最有压力的问题,报告人能立刻给出想清楚的答案,而不是当场慌乱。
3. **诚实写在论文/slide 里的范围声明**("32 状态、5 种子、原型规模"),让追问 2 的回应不需要现场
   编造任何托词,只是把已经想清楚、写清楚的内容再讲一遍。
4. **提前做过的 baseline 可复现性调研**(项目真实文档 `07-baseline-reproducibility-audit.md`),让
   追问 3 能给出一个具体、可信、不心虚的现状说明,而不是被问到才第一次意识到"我们好像没有真的对比
   过"。

**这是本篇 capstone 想强调的核心方法论**:一场看起来"临场应对得体"的 Q&A,几乎全部功劳应该记在
"报告之前的准备"上,而不是"报告人天生反应快"——04 号文件反复出现的"如实说明局限"这条原则,越是
提前在论文/slide/自我核验阶段就想清楚,现场能给出的回应就越从容;越是心存侥幸不去想清楚,现场就
越容易被逼到墙角,这不是运气问题。

---

*上一篇:[05-conference-attendance-guide.md](05-conference-attendance-guide.md) ·
返回 [00-roadmap.md](00-roadmap.md)*
