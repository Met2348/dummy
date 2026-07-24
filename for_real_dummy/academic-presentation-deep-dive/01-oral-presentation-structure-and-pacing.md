# 01 · Oral Presentation 结构与讲述节奏

> 这篇讲"一场 10-20 分钟的会议报告应该怎么组织",不是讲"怎么把 PPT 做好看"(那是
> [02-slides-design-principles.md](02-slides-design-principles.md) 的事)。核心矛盾:论文是写给"可以
> 回看、可以精读、有耐心啃公式"的读者的,talk 是讲给"只能听一遍、走神了就再也追不回来、忍不了枯燥"的
> 听众的——同一份工作,书面论证节奏和口头讲述节奏必须是两套完全不同的东西。这正是为什么 Simon Peyton
> Jones(SPJ,Haskell 语言核心设计者、Microsoft Research)会把"How to Write a Great Research Paper"和
> "How to Give a Great Research Talk"做成两份完全独立的讲座,而不是同一份材料换个格式——参见他的个人主页
> [simon.peytonjones.org/great-research-talk](https://simon.peytonjones.org/great-research-talk/) 和
> [Microsoft Research 录像页](https://www.microsoft.com/en-us/research/video/how-to-give-a-great-research-talk-4/)。

**格式说明**:本篇及本系列 02-05 号文件统一采用下面这套"六步演讲判断力模板"(改写自本仓库
[research-writing-deep-dive](../research-writing-deep-dive/00-roadmap.md) 系列为科研写作设计的
"六步写作判断力模板"——把原模板第⑤步"审稿人会怎么挑刺"换成"听众/评委会怎么问",因为这里的读者
面对的不是几周后写书面意见的审稿人,是几秒钟内必须张嘴回应的现场听众):

1. 常见误区/反面例子
2. 逐处修改对照
3. 可操作检查清单
4. 量化验证(真实代码)
5. 听众/评委会怎么问
6. 常见坑

场景素材说明:本篇部分举例借用用户真实在研项目 `research/world-model-imagination-controller/`(世界模型
测试时想象预算自适应分配控制器,目标 ICLR)的真实论证结构做场景灵感,不是替用户杜撰这篇论文的内容,细节
经过简化改编,不构成对该项目最终结论的代述。

---

## 1. 一个 talk 只讲一个 idea,不是压缩版论文

### 常见误区/反面例子

新手准备 talk 最常见的思路是"论文写了什么,我就照着讲一遍,只是把字数压缩"——摘要变成第 1 页,
Introduction 变成第 2-3 页,Method 全部小节各来一页,实验表格原样搬上去,Limitation 和 Future Work
各占一页。20 分钟的口头报告,被切成了论文目录的复读机。

SPJ 在这份讲座里把这个问题讲得很直接:talk 的目的不是"证明你做了很多工作"或者"复述论文",而是
**engage 听众、让他们想去读你的论文**——这是完全不同的目标函数。如果目标是"讲清楚论文里的每一个细节",
最终结果通常是听众在第 5 分钟就跟丢了,后面 15 分钟你在自说自话。

### 逐处修改对照

**改前(论文目录复读机式大纲,20 分钟报告)**:

```
1. Title (30s)
2. Abstract 逐句念一遍 (1min)
3. Introduction:问题背景+四段式动机+贡献列表 (4min)
4. Related Work:12 篇引用逐条介绍 (3min)
5. Method:公式推导全过程,含每个符号定义 (6min)
6. Experiments:5 张结果表格逐格解读 (4min)
7. Limitations (1min)
8. Conclusion + Future Work (30s)
```

**改后(围绕"一个 idea"重新分配)**:

```
1. Title + 一句话讲清楚"如果你只记住一件事"是什么 (1min)
2. Motivation:为什么现在的方法都固定分配想象预算,这有什么代价 (2-3min)
3. 核心 idea,用一个具体例子讲透——不是抽象描述,是带着听众走一遍"如果你是这个 controller,
   你现在看到了什么、要决定什么" (8-10min,全场重心)
4. 一个最有说服力的结果(不是全部结果),证明这个 idea 真的成立 (3min)
5. 我们和最近邻工作的一句话区别(不是逐篇 related work) (1min)
6. Take-home message 收尾,呼应开场那句话 (1min)
```

**每处为什么改**:
- Introduction 从"四段式学术套话"改成"直接讲清楚这个领域现在的真实浪费有多大"——比如"现有 world model
  的想象预算几乎都是训练前定死的常数,从 2018 年的 World Models 到 2026 年的最新工作,这个模式贯穿了
  8 年文献",一句话就能建立起"这个问题值得解决"的紧迫感,不需要摘要里的四段式铺垫。
- Related Work 从"逐篇介绍"压缩成"一句话区别"——SPJ 明确建议**不要在 talk 里专门讲 related work**,
  但你自己必须完全掌握它,这样才能在 Q&A 被问到"这和 XX 工作有什么区别"时随时接得住(这一条直接呼应
  [04-live-qa-skills.md](04-live-qa-skills.md) 的现场应对训练)。
- Method 的公式推导全过程被砍掉,换成"用一个具体例子带着听众走一遍决策过程"——技术细节留给论文本身,
  talk 里放的是能让人"看懂在做什么"的那一层,不是能让人"自己复现"的那一层。
- 时间被重新分配到"核心 idea"这一项上,占比从原来的 30%(6/20 分钟)提升到 40-50%——这不是随便定的
  比例,下一节会给出量化验证。

### 可操作检查清单

- [ ] 如果只允许听众记住一句话,这句话你能不能在开场 30 秒内讲出来?
- [ ] Related Work 有没有被压缩成"一句话区别",而不是逐篇介绍?(逐篇介绍留给论文和 Q&A)
- [ ] 公式/推导细节是不是只留了"听众能跟上"的那一层,完整推导留给论文?
- [ ] 核心 idea 部分是不是用了至少一个具体例子(不是纯抽象描述)?
- [ ] 结尾有没有回到开场那句话,形成闭环?

### 量化验证(真实代码)

"核心 idea 应该占大头时间"不是一句空话,可以用一个简单的"talk 时间预算审计工具"量化检查——按
120-140 词/分钟的口语语速估算大纲每部分要讲多久,再检查核心 idea 部分占比是否达标。这个语速区间来自
真实调研到的公开建议(见文末来源)。

```python
def estimate_minutes(word_count, wpm=140):
    return word_count / wpm

draft_bad = {
    "motivation": 150,
    "related_work_detailed": 600,
    "method_full_derivation": 700,
    "one_big_idea": 400,
    "results_table_dump": 500,
    "conclusion": 150,
}
draft_good = {
    "motivation": 250,
    "one_big_idea_with_examples": 950,
    "related_work_one_liner": 80,
    "conclusion_takehome": 120,
}

BUDGET_MIN = 12
WPM = 140

def audit(draft, label):
    total_words = sum(draft.values())
    minutes = estimate_minutes(total_words, WPM)
    idea_words = sum(v for k, v in draft.items() if "idea" in k)
    idea_share = idea_words / total_words
    over_budget = minutes - BUDGET_MIN
    print(f"{label}: total_words={total_words} est_minutes={minutes:.2f} "
          f"idea_share={idea_share:.2f} over_budget_by={over_budget:.2f}min")
    return minutes, idea_share

bad_minutes, bad_share = audit(draft_bad, "bad draft")
good_minutes, good_share = audit(draft_good, "good draft")

assert bad_minutes > BUDGET_MIN + 3, "坏范例应该明显超时"
assert bad_share < 0.3, "坏范例里核心idea占比应该明显偏低"
assert good_minutes <= BUDGET_MIN + 1, "好范例应该基本卡在预算内(留1分钟余量给现场语速波动)"
assert good_share >= 0.5, "好范例里核心idea+举例应该占多数时间(呼应SPJ的80%经验值方向,不强行卡精确80%)"
print("ALL TALK-BUDGET ASSERTIONS PASSED")
```

**如实说明这个工具的局限**:这只是"大纲阶段的字数预算审计",不是"真的会讲成这个时长"的保证——真实语速
因人而异、因紧张程度而异(这也是为什么下面第 4 节要求真正掐表排练,不能只信这个静态审计)。120-140
词/分钟是多个演讲训练资料反复出现的中间估计值,不是物理定律,自己的真实语速需要用第 4 节的方法实测。

### 听众/评委会怎么问

复用 dsa-deep-dive/statistics-deep-dive 已验证的"五轴追问链"方法论,这里给出**现场口语版**——审稿人
的追问隔着几周书面时间,听众的追问是几秒钟内当场发生的,同一套追问逻辑,应对的从容程度完全不同:

| 轴线 | 书面审稿版(系列 1 rebuttal 场景) | 现场口语版(这里) |
|---|---|---|
| 规模递增轴 | "小数据集上的结论能不能扩展到大数据集" | "这个例子你光讲了一个 idea,能泛化到别的场景吗?" |
| 决策依据追问轴 | "为什么选 A 方法不选 B 方法" | "你砍掉的那部分 related work,如果我现在问,你答得上来吗?" |
| 真实性验证轴 | "摘要这个说法有没有数据支撑" | "你刚才说的那个例子是真实跑出来的还是你现场编的?" |

对本节而言,最典型的现场追问是:"你这个 talk 完全没提 XX 相关工作,是不是不知道这篇?"——这正是为什么
"related work 从 talk 里砍掉"这个决定,前提是**你自己必须比听众更熟悉这些相关工作**,砍掉的是"讲给
听众听的时间",不是"自己该做的功课"。答不上来会让人怀疑你对领域的掌握程度,这比"talk 里提一下"更伤。

### 常见坑

- **把"减少 slide 数量"和"减少内容量"搞混**:把 8 页压缩成 4 页,但每页塞的字没变,不是真的做了减法,
  只是换了个更拥挤的排版(下一篇 [02-slides-design-principles.md](02-slides-design-principles.md) 专门讲
  这个陷阱)。
- **误以为"讲得越完整越显得工作扎实"**:企图在 20 分钟里塞进论文的每一个消融实验,后果通常是听众在
  中段彻底走神,反而没记住任何一条真正重要的结论——SPJ 原话精神是"talk 的作用是让人想读你的论文",
  不是替代论文,该省的细节留给论文和 Q&A,不是不负责任地藏起来。
- **忘了"talk 不是用来读的"**:把讲稿写成一篇可以逐字照读的文章,现场变成"念稿子"——听众自己会读字,
  你的价值在于讲稿之外你要补充的东西(一个类比、一段故事、一次现场追问式的自问自答)。这条会在
  [02-slides-design-principles.md](02-slides-design-principles.md) 里从"slide 该放多少字"这个角度继续展开。

---

## 2. 时间怎么分配:motivation / idea / 细节 / related work 的真实比例

### 常见误区/反面例子

常见的错误分配方式是"平均主义"——20 分钟报告,Introduction、Method、Experiments、Discussion 四大块
各分 5 分钟,看起来"公平",但公平不等于有效。真正决定听众能不能记住东西的,不是"每个部分讲了多久",
是"核心论点讲透了没有"。

### 逐处修改对照

**改前**:Introduction 5min / Method 5min / Experiments 5min / Discussion+Q&A 预留 5min——四等分。

**改后**:Motivation 2-3min / 核心 idea(含例子)8-10min / 最有说服力的一组结果 3min / 一句话
related work 区别 1min / Take-home 收尾 1min——把大头时间显式砸在"核心 idea"这一项,其余部分
只服务于让核心 idea 立得住。

**为什么改**:这不是拍脑袋决定的比例,是 SPJ 讲座反复强调的分配原则——**talk 的大部分时间应该花在
一个核心 idea 上,用具体例子讲透,而不是平均分给论文的每个部分**。这和写论文时 Introduction/Method/
Experiments 相对均衡的篇幅分配是两套完全不同的逻辑:论文的读者可以按自己的节奏跳读、回看、只看感兴趣
的部分;talk 的听众只能跟着你的时间线走一遍,均分时间意味着没有任何一部分被真正讲透。

### 可操作检查清单

- [ ] 核心 idea 部分是不是全场时间占比最大的一块(不是并列最大,是明显最大)?
- [ ] Motivation 有没有控制在 2-3 分钟内讲完"为什么这个问题重要",而不是变成第二个 Introduction?
- [ ] 是不是只选了"最能证明核心 idea 成立"的一组结果,而不是把论文里所有表格都搬上来?
- [ ] 时间分配是不是留了机动余量(比如 12 分钟的槽位准备 10-11 分钟的内容),而不是卡着上限设计?

### 量化验证(真实代码)

复用第 1 节验证过的 `estimate_minutes` / `audit` 逻辑,这里换一个角度验证:给定一个总时长预算,
"平均分配四块"和"idea 占大头"两种分配方式,分别检查是否满足"核心内容时间 ≥ 总预算一半"这条底线。

```python
def minutes_from_share(total_minutes, shares):
    assert abs(sum(shares.values()) - 1.0) < 1e-6, "各部分占比之和必须是100%"
    return {k: total_minutes * v for k, v in shares.items()}

TOTAL = 15  # 一个常见的会议 oral 时长量级(具体每个会议年年不同,这里只做比例演示)

even_split = {"motivation": 0.25, "idea": 0.25, "results": 0.25, "related_work_and_qa_buffer": 0.25}
idea_heavy = {"motivation": 0.18, "idea": 0.58, "results": 0.17, "related_work_and_qa_buffer": 0.07}

even_minutes = minutes_from_share(TOTAL, even_split)
idea_minutes = minutes_from_share(TOTAL, idea_heavy)

print("even split:", {k: round(v, 1) for k, v in even_minutes.items()})
print("idea-heavy split:", {k: round(v, 1) for k, v in idea_minutes.items()})

assert even_minutes["idea"] < TOTAL * 0.5, "平均分配下idea部分达不到过半时间,如实复现'常见误区'里的问题"
assert idea_minutes["idea"] >= TOTAL * 0.5, "idea占大头的分配方式应该让核心内容真正过半"
print("ALL TIME-ALLOCATION ASSERTIONS PASSED")
```

### 听众/评委会怎么问

- **决策依据追问轴**:"你花了快一半时间在讲这一个例子,为什么不多讲几个实验结果?" ——诚实的回答方向
  是"因为这个例子如果没讲透,后面的结果数字对没听懂机制的人是没有意义的;结果数字本身论文里都有,今天
  我想确保走出这个房间的人真的理解了这个 idea 在做什么"。
- **真实性验证轴**:"你说这是'最有说服力的结果',为什么不是论文里最好看的那张表?" ——如果这组结果确实
  不是数字最大的那组,而是"最能证明核心机制成立"的那组,要能诚实讲清楚这个选择标准,而不是被追问到才
  含糊其辞。

### 常见坑

- **把"预留 Q&A 时间"算漏**:20 分钟的槽位,很多会议实际上是 15 分钟讲 + 5 分钟问,把 20 分钟内容
  硬塞满,现场就没有 Q&A 时间,主持人会直接打断——预算时间时永远要按"槽位时长 − Q&A 时长"来算,不是
  按槽位总时长。
- **"机动余量"没有落到实处**:知道要留余量,但排练时从来没有真的卡表,导致真实上台时因为紧张语速变化
  (更快或更慢)而完全打乱计划——这条直接对应第 4 节的排练方法论,光有静态时间预算是不够的。

---

## 3. 开场 2 分钟与收尾的 take-home message

### 常见误区/反面例子

**开场常见误区**:上台先说"大家好,我是 XXX,来自 XXX 实验室,今天要讲的是……",然后照着 Title
slide 把论文标题念一遍,再花 1 分钟感谢合作者——听众在这 1-2 分钟里完全不知道"为什么要听下去"。

**收尾常见误区**:讲完最后一张实验结果的 slide 后直接说"谢谢大家,以上就是我的报告",没有任何总结,
听众带走的最后印象是一张密密麻麻的表格,不是这个 idea 本身。

### 逐处修改对照

**改前(开场)**:"大家好,我叫 XXX,来自 XXX 组,今天给大家汇报一下我们关于世界模型想象预算分配的
工作,首先看一下 outline……"

**改后(开场)**:"现有的世界模型,不管想多久、想几个候选,这个预算几乎都是训练前定死的常数——今天我
想说服大家,这笔预算浪费得有多离谱,以及一个决定'什么时候该多想、什么时候不该'的简单原则。"

**为什么改**:第一句话就要建立"为什么现在要听你讲"的理由,而不是走流程式的自我介绍——SPJ 的建议是把
"钩子"放在最前面,而不是放在 outline 之后。这不代表完全不能自我介绍,而是自我介绍不能占用听众注意力
最集中的开场时刻。

**改前(收尾)**:最后一张 slide 是"Table 5: Ablation Results",讲完直接"谢谢"。

**改后(收尾)**:回到开场那句话,专门用一张 slide 呼应:"回到开始的问题——想象预算不该是常数,而是
应该看这次想象有没有真正带来基线不知道的信息。这是今天想让大家带走的一句话。"

**为什么改**:收尾是听众记忆最后被强化的时刻,如果这里放的是一张数据表格,听众带走的印象就是数据表格
本身,而不是背后的道理。呼应开场的收尾能在听众脑子里形成一个完整的闭环故事,而不是戛然而止。

### 可操作检查清单

- [ ] 开场第一句话是不是"钩子"(为什么值得听下去),而不是流程式的自我介绍或 outline?
- [ ] 收尾有没有一句"如果只记住一件事"式的 take-home message,而不是停在最后一张数据 slide?
- [ ] 开场和收尾是不是在呼应同一个问题/同一句话,形成闭环?
- [ ] 有没有为可能超时的情况准备"如果只剩 2 分钟该跳到哪"的应急方案?(SPJ 明确提到:不要为了讲完
      所有 slide 而牺牲收尾——收尾比中间任何一张 slide 都重要)

### 量化验证(真实代码)

"开场钩子"和"收尾 take-home"本身是判断力问题,不能被 assert,但可以做一个客观检查:一份好的收尾应该
包含和开场高度重合的关键词(形成呼应闭环),而不是完全脱节的新内容。这是一个真实可跑的、基于关键词
重叠度的启发式检查(不是完美的语义检查,只能抓字面重合,如实说明局限)。

```python
import re

def keywords(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    stopwords = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "this", "that", "we", "it"}
    return {w for w in words if w not in stopwords and len(w) > 2}

opening_good = "Imagination budgets are fixed constants -- I want to convince you that's a waste."
closing_good = "Coming back to the start: imagination budgets shouldn't be constants, that's the takeaway."

opening_bad = "Hi everyone, I'm from the XX lab, today I'll present our outline."
closing_bad = "Table 5 shows the ablation results. Thanks everyone."

def overlap_ratio(a, b):
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)

good_ratio = overlap_ratio(opening_good, closing_good)
bad_ratio = overlap_ratio(opening_bad, closing_bad)
print(f"good open/close keyword overlap = {good_ratio:.2f}")
print(f"bad  open/close keyword overlap = {bad_ratio:.2f}")

assert good_ratio > bad_ratio, "首尾呼应的版本,关键词重合度应该明显更高"
assert good_ratio >= 0.2, "好范例的首尾重合度应该达到一个有意义的下限"
print("ALL OPEN-CLOSE-LOOP ASSERTIONS PASSED")
```

### 听众/评委会怎么问

这一节的现场反馈通常不是"提问",而是更残酷的"沉默"——如果开场没有钩住听众,Q&A 环节问题会明显变少
或变得敷衍(比如只有主持人礼貌性地问一个问题)。**方案批判迭代轴**的现场版:如果你观察到往期同一个人
的几场 talk 都是"问题寥寥",这本身就是一个信号,值得回去检查开场设计,而不是归咎于"这次听众不感兴趣"。

### 常见坑

- **开场道歉**:"这个 talk 我准备得比较仓促""我的电脑刚才出了点问题"——SPJ 明确建议不要用这类道歉
  开场,听众不关心你准备得够不够充分,道歉只会让人对接下来的内容降低期待,于事无补。
  参见 [Microsoft Research 讲座页](https://www.microsoft.com/en-us/research/video/how-to-give-a-great-research-talk-4/)。
- **收尾突然被时间打断**:因为中间部分超时,收尾被压缩成"啊时间到了,谢谢大家"——这是第 2 节"预留
  机动余量"和第 4 节"真实排练掐表"没有做到位的直接后果,收尾这个环节本身值得被优先保护,而不是
  第一个被时间挤压掉的部分。

---

## 4. 排练方法论:语速换算与多轮排练

### 常见误区/反面例子

常见误区是"心里过几遍就行,正式讲的时候临场发挥"——问题是,不经过大声朗读+掐表的排练,几乎不可能
准确估计自己的真实时长,因为脑内默读的速度和真实开口讲话的速度是两回事,紧张时的语速波动更是无法靠
"心里过一遍"预判。

### 逐处修改对照

**改前**:准备好 slides 后,只在脑子里过一遍逻辑顺序,确认"内容都在",没有真正开口计时。

**改后**:
1. 先按语速公式估算总时长(见下方代码),粗略判断内容量是否合理。
2. 独自大声朗读、掐表、录音或录屏,先过 2-3 轮不追求流畅度,只是把"卡壳的地方"暴露出来。
3. 从第 3 轮开始才正式掐表计时(前几轮因为不熟练,计时没有参考价值)。
4. 找同学/同门做真实听众,面对人讲一遍——独自排练和有人在场排练的紧张程度完全不同,必须真实模拟。
5. 尽量用真实会用到的设备(投影/翻页笔/麦克风)走一遍,减少正式上台时的意外。

**为什么改**:这是真实调研到的排练建议共识——先建立流畅度、再开始计时(前几轮估计通常偏长,是正常
现象,不代表内容真的超量);超时了要真的删内容,而不是寄希望于"正式讲的时候会不自觉加快"。

### 可操作检查清单

- [ ] 有没有至少完整大声排练 3 轮以上(不是脑内默读)?
- [ ] 有没有至少一次是对着真人(不是对着空房间)排练的?
- [ ] 有没有用真实会用到的设备/环境走过一遍?
- [ ] 掐表结果如果超时,是不是真的删了内容,而不是"到时候再说"?
- [ ] 是否录音/录屏回看过至少一次,检查语速、口头禅、卡壳点?

### 量化验证(真实代码)

排练的核心是把"内容量"换算成"预计时长",再用真实掐表结果校正。下面这个工具做两件事:①用语速公式
估算时长;②模拟"多轮排练时长逐渐收敛"这个真实现象(前几轮偏长,趋于稳定后才是可信数字),用于说明
"为什么不能只排练一轮就下结论"。

```python
def estimate_minutes(word_count, wpm):
    return word_count / wpm

talk_word_count = 1500

# 语速会随排练轮次趋于稳定(紧张感下降+越来越熟悉内容),这里用一组真实量级的模拟数据演示这个模式,
# 不是编造一个"看起来很整齐"的数列——刻意让前几轮明显偏离、后几轮趋于一致
rehearsal_wpm_by_round = [95, 110, 128, 135, 138, 140, 139]

durations = [estimate_minutes(talk_word_count, wpm) for wpm in rehearsal_wpm_by_round]
for i, d in enumerate(durations, 1):
    print(f"round {i}: {rehearsal_wpm_by_round[i-1]}wpm -> est. {d:.2f} min")

first_round, last_three = durations[0], durations[-3:]
last_three_spread = max(last_three) - min(last_three)
first_vs_last_gap = abs(first_round - sum(last_three) / 3)

assert first_vs_last_gap > 1.0, "第一轮和后几轮的估计时长应该有明显差距(佐证'第一轮不能作数')"
assert last_three_spread < 0.5, "后几轮应该已经收敛到比较接近的时长(佐证'排够轮次后数字才可信')"
print("ALL REHEARSAL-CONVERGENCE ASSERTIONS PASSED")
```

**如实说明**:这组语速数字是按"真实排练中紧张感逐轮下降"这个普遍现象构造的合理模拟,不是某一次真实
排练的原始记录——用来演示"为什么第一轮计时不可信、要看后几轮收敛值"这个方法论,不是给出一个万能的
语速数值表,自己的真实语速必须自己实测。

### 听众/评委会怎么问

排练不足的现场信号是最直接的:讲到一半突然停顿找词、结尾被主持人打断、语速全程偏快像在赶进度——这些
都不是"临场发挥失常",是排练轮次不够的必然结果。**这一节没有对应的"听众提问"**,因为排练是报告之前
的准备工作,听众感知到的是排练不足的后果,而不是对排练本身提问——如实说明这一点,而不是硬凑一个不
自然的"听众会怎么问"。

### 常见坑

- **只掐表不录音/录屏**:掐表能知道"讲了多久",但看不到"哪里在重复口头禅、哪里语速突然变快"——这些
  只有回看录音/录屏才能发现,是排练里最容易被跳过、但回报很高的一步。
- **最后一次排练留到上台前一晚**:前几轮排练本来就应该讲得磕磕绊绊,这是正常过程,不是"没准备好"的
  信号——但如果只留一晚时间,会把这个本该分散在几天里的正常过程压缩成一次性的高压尝试,体验很差,
  也来不及根据回看发现的问题做调整。

---

## 参考来源

- Simon Peyton Jones, *How to Give a Great Research Talk*,个人主页
  [simon.peytonjones.org/great-research-talk](https://simon.peytonjones.org/great-research-talk/)
  (与 *How to Write a Great Research Paper* 并列的独立姊妹讲座)
- Microsoft Research 讲座录像页
  [microsoft.com/en-us/research/video/how-to-give-a-great-research-talk-4](https://www.microsoft.com/en-us/research/video/how-to-give-a-great-research-talk-4/)
- 排练方法论(多轮排练、录音回看、模拟真实条件、语速与词数换算)综合自多篇学术演讲排练指南,
  检索关键词 "how to rehearse a scientific presentation" "conference presentation rehearsal checklist"。

---

*上一篇:无(本系列首篇) · 下一篇:[02-slides-design-principles.md](02-slides-design-principles.md)*
