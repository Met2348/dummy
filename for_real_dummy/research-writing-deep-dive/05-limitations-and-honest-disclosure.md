# 05 · 局限性/风险自曝的诚实写法(Limitations & Honest Disclosure)

> 总览见 [00-roadmap.md](00-roadmap.md)。这一篇讲"不自毁但也不回避"——局限性写作是整个系列里判断力
> 要求最高的一类内容,过头和不足都会实实在在地伤害论文,而且伤害的方向完全相反。

---

## 1. 两种失败模式:hedge 太多和彻底回避,拉着相反的方向

局限性写作有一个不太直观的事实:它可能以**两种完全相反的方式**失败,而不是只有"写得不够"这一种
风险。一种是**过度 hedge**——每句话都堆满"may"/"might"/"could"/"to some extent"/"in some cases"
这类模糊限定词,读者读完会开始怀疑"这个研究到底有没有发现任何站得住的东西";另一种是**彻底回避**——
只字不提读者一眼就能看出的明显弱点,或者用"despite these limitations, our findings remain very
strong"这种空洞的自我肯定草草收尾。这两种失败模式拉着相反的方向,意味着"写局限性"不是一个"越保守
越安全"的单调问题——过度保守本身就是一种真实的失败模式。

**常见误区/反面例子(过度 hedge 版本):**

> Our method may not generalize well in some cases and could have some limitations that might
> affect performance to some extent under certain scenarios.

24 个词里堆了 6 处模糊限定,读完完全不知道具体是什么局限、影响到底有多大——这种写法与其说是"诚实
自曝",不如说是"用模糊语言规避承诺任何具体信息",反而会让审稿人怀疑作者自己也说不清楚问题出在哪。

**逐处修改对照:**

> We validated the controller only in a 6x6 tabular gridworld with 32 states and a learned
> frequency-based world model; we have not tested whether the same conclusions hold for
> high-dimensional latent world models such as DreamerV3, where the imagination rollout and
> the baseline critic are far less likely to share exactly the same model.

同样是承认局限,但换成了具体的场景描述(6x6 gridworld、32 states)、具体的对比对象(DreamerV3)、
具体的机制解释(为什么高维潜空间模型下"共享同一个模型"这个前提可能不成立)。

**可操作检查清单:**
- [ ] 通读局限性段落,统计模糊限定词("may"/"might"/"could"/"to some extent"/"in some cases")
  的密度——密度过高说明可能陷入了"用模糊语言回避具体信息"这个反模式
- [ ] 是否有至少一处具体的范围声明(具体数据集/具体规模/具体设置),而不是清一色的模糊限定
- [ ] 结尾是否有"despite these limitations, our findings remain very strong/important"这类空洞
  的自我肯定——如果有,考虑换成更有分寸的措辞(见知识点 4)

**量化验证:**

```python
import re

HEDGES = ["may not", "might not", "could", "possibly", "to some extent", "in some cases",
          "some limitations", "certain scenarios", "may have", "might have"]

def hedge_density(text):
    lower = text.lower()
    words = re.findall(r"[A-Za-z']+", text)
    hits = sum(len(re.findall(re.escape(h), lower)) for h in HEDGES)
    has_number = bool(re.search(r"\d", text))
    return {"words": len(words), "hedge_hits": hits,
            "hedge_per_100w": hits / max(len(words), 1) * 100,
            "has_concrete_number": has_number}

vague_limits = (
    "Our method may not generalize well in some cases and could have some limitations "
    "that might affect performance to some extent under certain scenarios."
)

specific_limits = (
    "We validated the controller only in a 6x6 tabular gridworld with 32 states and a "
    "learned frequency-based world model; we have not tested whether the same conclusions "
    "hold for high-dimensional latent world models such as DreamerV3, where the imagination "
    "rollout and the baseline critic are far less likely to share exactly the same model."
)

r_vague = hedge_density(vague_limits)
r_specific = hedge_density(specific_limits)
assert r_vague["hedge_per_100w"] > 0 and r_vague["has_concrete_number"] is False
assert r_specific["hedge_per_100w"] == 0.0 and r_specific["has_concrete_number"] is True
print("vague_limits (bad example):", r_vague)
print("specific_limits (revised):", r_specific)
```

本机实测:`vague_limits` 24 词命中 6 处 hedge(密度 25.0/百词),没有任何具体数字;`specific_limits`
55 词、0 处 hedge、包含具体数字(6x6、32)。**明确边界**:hedge 密度为 0 不是绝对目标——完全不用任何
限定词反而可能过度自信,判断力要落在"这段话有没有提供具体到可以核查的信息",不是单纯追求 hedge 词
数量最小化。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你说'我们只在 32 状态的合成环境验证过',这个规模选择本身是不是也该说明原因,
  不然读起来像是'能力不够而不是刻意选择'?" → 反驳:确实应该补充——诚实的局限性陈述不只是承认
  "做了什么范围",还应该说明"为什么选这个范围"(比如需要 ground-truth 才能做诊断性验证,合成环境
  能精确算出最优解,真实环境做不到),这样读者能分清"这是方法论上的刻意取舍"还是"资源不足的妥协"。

**常见坑:**
1. 只在结尾补一段笼统的"局限性讨论",却没有在正文对应的具体位置提前埋下伏笔——好的局限性讨论应该
   让读者读到具体结果时就有心理预期,不是读到最后才发现"原来前面那个数字有这么大的保留条件"。
2. 用被动语态弱化局限性的表述("some limitations may exist")——这和
   [04 类](04-sentence-level-academic-english.md)讲的被动语态问题是同一个机制,在局限性段落里
   尤其容易被滥用成回避主语("是谁的局限、谁没做到")的工具。

---

## 2. NeurIPS Checklist 的态度:诚实不扣分,被审稿人发现才扣分

NeurIPS 官方的 Reviewer Guidelines 和 Paper Checklist 指南对局限性披露的态度非常明确,而且和很多
作者的直觉相反:**"answering 'no' to some questions is typically not grounds for rejection"**——
审稿人被明确要求"不能因为作者诚实回答'没有做到'就据此拒稿"。官方指南进一步解释了这条规则背后的
逻辑:担心"完全诚实地承认局限性会被审稿人抓住当拒稿理由"这种恐惧本身是可以理解的,但**更糟的结果
是审稿人自己发现了论文没有承认的局限性**——那种情况下,读者/审稿人的第一反应不只是"这篇论文有个
局限",而是"作者是不是故意隐瞒了这一点",信任一旦破裂,对整篇论文的观感会一起受损。这也是为什么
checklist 里区分 NA(no limitation to discuss)和 No(有局限但没有在文中讨论)——后者才是真正的
问题,而不是"有局限性"这件事本身。

**常见误区/反面例子:** 因为担心"承认局限性=送人口实",干脆完全不写 Limitations 一节,或者把它压缩
成一句"we leave further exploration to future work"这种没有实质内容的收尾——这种做法在 checklist
制度下反而是更危险的选择:被审稿人自己发现未披露的局限,后果比主动披露严重得多。

**可操作检查清单:**
- [ ] 是否因为"担心被拒"而回避写清楚某个明显的局限——如果一个有经验的审稿人一眼就能看出这个局限,
  主动写出来永远比被发现后动摇整体信任更安全
- [ ] Checklist 类问题(如"是否讨论了局限性")如果诚实的答案是"No"（有局限但没讨论）,是否已经
  补充了对应的讨论,而不是寄希望于审稿人不会深究
- [ ] "future work"是否写得具体(比如"扩展到 DreamerV3 需要重新定义信息通道边界,这是一个具体的
  技术问题")而不是一句模板化的收尾

**量化验证:** 这一条是官方政策事实和写作心态问题,不是可以用代码验证的陈述——"诚实披露是否真的
不会被扣分"取决于具体审稿人是否遵守官方指南,这是判断力/信任问题,不硬造一个假的验证脚本。可以
诚实标注的是:这条规则本身是可查证的公开事实(NeurIPS 2024/2025 Reviewer Guidelines、Paper
Checklist Guidelines 官方页面),不是本系列的主观建议,读者可以自行核查目标 venue 当年的官方政策
是否有相同条款——不同会议/期刊的具体规定可能不同,投稿前应确认目标 venue 当年的规则,不能想当然
套用 NeurIPS 的规则。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"官方说'诚实不扣分',但审稿人也是人,真的能保证不受影响吗?" → 反驳:官方
  指南确实是理想状态下的约束,不能保证每个审稿人都严格执行,但两权相害取其轻——主动披露至少给了
  审稿人"作者知道这一点、并且在负责任地讨论它"这个印象,被动等着被发现,唯一可能的结果只会更差,
  不会更好。

**常见坑:**
1. 把"诚实披露不扣分"这条规则当作"局限性写得越多越好"的许可——过度堆砌不相关或过度自我否定的
   局限性(比如把方法论上无关紧要的细节也写成"局限"),同样会稀释真正重要局限的分量,不是诚实
   越多越安全,是**该披露的都披露、不该混进来的噪声不要混进来**。
2. 没有意识到不同 venue 的态度可能不同——有些审稿人群体确实会对诚实披露的局限性反应过度,这是
   真实存在的风险,官方指南是约束审稿人的规则,不是保证审稿人一定会遵守。

---

## 3. 具体 vs 泛泛:命名一个真实的、可能存在的混淆变量,比笼统道歉更可信

一个具体、可信的局限性陈述应该**点名一个真实、具体的潜在混淆因素**,并且诚实说明它可能往哪个方向
偏移结论;笼统地说"其他因素可能影响了结果"这种说法几乎没有信息量,因为它没有承诺任何可以被验证的
具体内容,反而更像是一种形式主义的"免责声明"。

**常见误区/反面例子:** "Other factors may have influenced the results, and further work is needed
to fully understand the phenomenon."——这句话可以原封不动地贴在任何一篇论文的任何一个结果后面,
因为它没有指出任何具体的东西。

**逐处修改对照:** 把"其他因素"替换成一个具体的、读者可以自己判断是否重要的候选混淆变量:"The
32-state gridworld is small enough that value iteration converges to a near-exact optimum; in
larger or partially observed environments, the baseline critic itself may be far less accurate,
which could change the sign of the comparison between imagination and no-imagination."——这句话
点名了一个具体机制(baseline critic 的准确度)、说明了它可能怎样影响结论(甚至可能反转符号),读者
读完能自己判断这个担忧有多严重。

**可操作检查清单:**
- [ ] 每一条局限性陈述是否点名了一个具体的机制/变量,而不是"其他因素"这种无法验证的笼统说法
- [ ] 是否说明了这个混淆因素可能往哪个方向影响结论(高估/低估/甚至反转符号),而不是只说"可能有影响"
- [ ] 具体范围声明(数据集规模、模型架构、样本量)是否用了确切数字,而不是"较小的规模"这种相对表述

**量化验证:** "混淆变量的解释是否合理"是判断力问题,但"陈述里有没有具体范围数字、有没有用模糊
量词兜底"是可以检查的表层信号:

```python
import re

VAGUE_QUANTIFIERS = ["some cases", "certain scenarios", "several factors", "many settings",
                      "various conditions", "a number of situations"]
SPECIFIC_SCOPE_RE = re.compile(r"\b\d+(\.\d+)?\s?(states?|seeds?|x\d+|episodes?|tasks?|dimensions?)\b",
                                re.IGNORECASE)

def specificity_check(text):
    lower = text.lower()
    vague_hits = sum(1 for v in VAGUE_QUANTIFIERS if v in lower)
    specific_hits = len(SPECIFIC_SCOPE_RE.findall(text))
    return {"vague_quantifier_hits": vague_hits, "specific_scope_mentions": specific_hits}

vague = "Our method has some limitations in certain scenarios and may not work in several factors."
specific = ("We validated the controller only in a 6x6 tabular gridworld with 32 states over "
            "5 seeds; generalization to higher-dimensional tasks is untested.")

r1 = specificity_check(vague)
r2 = specificity_check(specific)
assert r1["vague_quantifier_hits"] >= 2 and r1["specific_scope_mentions"] == 0
assert r2["specific_scope_mentions"] >= 2
print("vague (bad example):", r1)
print("specific (revised):", r2)
```

本机实测:`vague` 命中 2 处模糊量词、0 处具体范围;`specific` 命中 3 处具体范围提及("6x6"里的 6、
"32 states"、"5 seeds")、0 处模糊量词。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"你点名了'baseline critic 在大环境下可能不准确'这个混淆因素,但没有给出任何
  证据这个担忧有多大——这算不算又一种变相的模糊?" → 反驳:承认一个具体机制的存在、和量化它的影响
  大小,是两个不同深度的要求——如果确实没有资源去测量这个影响的具体大小,诚实地"点名机制但不量化
  影响"依然比完全不提要好,但如果审稿人认为这个未量化的担忧已经严重到动摇论文的核心结论,这就是
  一个需要在 rebuttal 阶段([08 类](08-rebuttal-writing-techniques.md))认真处理的问题,不能永远
  停留在"我提到过这个风险"这个层面。

**常见坑:**
1. 点名了具体机制,但只往对自己有利的方向解读("这个因素在我们的场景下影响应该很小")——诚实的
   点名应该包含"最坏情况下这会怎样"的判断,不能只挑对自己有利的解读。
2. 具体数字和正文其他地方的数字不一致(比如局限性一节说"5 seeds",正文实验部分实际用了 3 seeds)——
   这种前后矛盾比笼统模糊更容易被审稿人当场抓到,任何具体数字都要和正文交叉核对。

---

## 4. 局限连接到"对结果解读的影响",而不是孤立列出

一份常见的弱局限性写法是把局限性列成一份和正文脱节的清单——读者读完知道"这些是局限",但不知道
"这些局限具体应该怎么影响我对前面那些结果的信任程度"。更有效的写法是**每条局限后面直接跟一句
"这意味着……"**,把局限和"结果该被如何解读"显式连接起来,让局限性讨论真正服务于帮助读者校准
信任程度,而不是走一个形式主义的流程。

**常见误区/反面例子:**

> Our results come from a synthetic gridworld with a tabular model. We also only tested three
> target settings.

两句话孤立地陈述了两个事实性限制,完全没有说"所以呢"——读者不知道这两条限制应该让他们对前面报告的
82.0% 这个数字打几折信任。

**逐处修改对照:**

> Our results come from a synthetic gridworld with a tabular model; readers should not read
> the 82.0% hit rate as a claim that generalizes to high-dimensional latent world models,
> since we have not tested that setting.

同样的事实,加上了一句明确的解读指引:这个数字是关于"合成环境下的一个存在性证明",不是关于"任意
world model 都成立"的泛化主张。

**可操作检查清单:**
- [ ] 每条局限性陈述后面,是否有一句话明确指出"这应该怎样调整读者对前文结果的信任/理解范围"
- [ ] 是否明确区分了"这个结果是存在性证明(某种情况下这确实发生过)"还是"普遍性主张(在所有相关
  情况下都成立)"——很多论文的核心结论其实只是前者,但摘要/Introduction 的措辞容易不小心暗示成后者
- [ ] 结尾是否避免了"despite these limitations, the study is still very important"这类空洞的
  自我重新肯定——更站得住的收尾方式是给出一个校准过的、有分寸的总结("在这些约束条件下,这项工作
  提供了……的早期证据,有待后续工作在……场景下检验")

**量化验证:** "解读指引是否合理"需要判断力,但"局限陈述后面是否存在一句显式的解读连接语"是可以
用信号词检测的:

```python
import re

IMPLICATION_SIGNALS = ["this means", "therefore", "as a result", "should be interpreted",
                        "we caution", "this suggests", "readers should", "should not be read as"]

def has_implication_link(paragraph):
    lower = paragraph.lower()
    return any(sig in lower for sig in IMPLICATION_SIGNALS)

isolated = ("Our results come from a synthetic gridworld with a tabular model. We also only "
            "tested three target settings.")

linked = ("Our results come from a synthetic gridworld with a tabular model; readers should "
          "not read the 82.0% hit rate as a claim that generalizes to high-dimensional latent "
          "world models, since we have not tested that setting.")

r1 = has_implication_link(isolated)
r2 = has_implication_link(linked)
assert r1 is False and r2 is True
print("isolated (bad example, listed with no interpretation link):", r1)
print("linked (revised, connected to how it affects interpretation):", r2)
```

本机实测:`isolated` 未命中任何解读连接信号(`False`),`linked` 命中"readers should"(`True`)。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你说读者不该把 82.0% 读成'泛化到高维模型的证据',那这个数字在论文里到底能
  用来支撑什么主张?" → 反驳:这正是显式写出解读连接语的价值——逼着作者自己想清楚"这个数字唯一
  站得住的主张是什么"(这里是"存在一种信息优势能让想象反超基线的场景"),而不是含糊地让读者自己
  去脑补一个可能过度泛化的主张。

**常见坑:**
1. 解读连接语写得比局限性本身更长、更啰嗦,反而稀释了局限性陈述本身的清晰度——这句连接语应该是
   精炼的一句话,不是又一段新的论证。
2. 只对"无关痛痒"的局限性写解读连接语,对真正核心的局限保持沉默——解读连接语的价值恰恰在于把它
   用在读者最需要被提醒的地方,不是用在容易写的地方。

---

## 5. 真实案例:一次"意外的坏消息"是怎么被诚实写出来的

**取材 `research/world-model-imagination-controller/02-deep-gap-analysis.md` §3(按设计文档要求
做教学化改写,重点是写法本身,不是复述项目结论):** 项目组在深入调研阶段发现,自己 pilot 阶段最初
认为的"发现"("想象和基线共享同一个不完美模型时,多算不会变好")其实是 Russell & Wefald(1991)
Value of Computation 理论的直接推论,已经被后续文献形式化过——**而这篇形式化论文本来就在自己的
文献库里**。这是一条会直接削弱"新颖性"这个核心卖点的坏消息,但项目组把它写进了文档里,而且用了
几个值得学习的具体语言选择:

1. **主动点出风险的具体后果,而不是含糊带过**:文档原话大意是"如果论文把核心贡献叙事锚定在这个
   发现本身,审稿人只要熟悉相关文献,几乎肯定会指出这是 reinventing the wheel,这正是 desk reject
   最典型的成因之一"——直接说出"如果不改会怎样",而不是轻描淡写地说"这条可能需要再考虑一下"。
2. **明确要求"现在改口径,不能等审稿人指出来才改"**——这句话把"诚实自曝"和"多轮修改方法论"
   ([06 类](06-revision-methodology-and-ai-boundary.md)会展开)直接联系起来:自我审查的时间点
   应该在投稿前,不是等审稿意见回来才第一次意识到问题。
3. **没有把这条坏消息处理成"这部分工作全部作废",而是重新定位了它的价值**——原本当"新发现"卖,
   现在改口径为"理论基线与阴性对照"(第一次在这个具体场景下用严格数学把已知原理做实、可复现），
   这是合法的贡献,只是评价标准变了,诚实讲清楚这个转变本身,反而向审稿人证明做过认真的文献调研。

**可操作检查清单:**
- [ ] 是否在写作/投稿前,主动去检验自己认为"新"的结论,是不是某个更早理论的已知推论——检验方式是
  回头系统翻自己已经收集的文献库,而不是假设"没读到过就等于不存在"
  (这条判断力和 [01 类](01-narrative-structure-and-elevator-pitch.md)知识点 4 講的"核实贡献是否
  真的新颖"是同一件事的不同角度,这里强调的是"怎么写出这个检验过程",那边强调的是"贡献列表本身该
  怎么收敛")
- [ ] 发现坏消息后,是否诚实评估了"如果不处理,审稿人大概率会怎么反应",而不是寄希望于审稿人没
  注意到
- [ ] 是否尝试给这条坏消息重新定位价值,而不是简单地删除或者假装没看见——很多"坏消息"换一个评价
  标准依然是合法贡献,只是不能继续用最初那套包装

**量化验证:** 这是一次真实的、发生在文档写作过程中的自我纠正,本质是判断力和诚实态度问题,没有
代码可以验证"这次自我纠正是否做得足够好"——如实标注这是本篇里最不可量化的一个知识点,量化工具在
局限性写作里只能覆盖表层的语言特征(hedge 密度、具体性、解读连接),覆盖不了"有没有真的去做那次
文献核查"这件事本身,这件事只能靠研究者自己的严谨态度。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你怎么证明真的做了这次'回头核查文献库'的工作,而不是投稿后被审稿人指出才补的?"
  → 反驳:诚实的答案是审稿人通常无法从最终论文文本本身区分"投稿前主动核查"和"投稿后被迫承认"——
  这正是为什么"现在改口径,不能等审稿人指出来才改"这条纪律本身有价值:不是为了让审稿人看出"我们
  多主动",而是因为提前核查能带来真实的好处(有时间重新定位价值、调整论证重心),等被指出来才补救,
  除了写一段防御性的 rebuttal,已经没有更好的选择了。

**常见坑:**
1. 发现坏消息后陷入"是不是这部分工作全废了"的过度悲观,直接砍掉而不尝试重新定位价值——很多"不再
   新颖"的结论依然是合法的"理论基线/阴性对照"贡献,砍掉可能是过度反应。
2. 反过来,发现坏消息后用更复杂的措辞把它包装得让人看不出这是个坏消息——诚实自曝的价值恰恰在于
   "让审稿人一眼看出你知道这件事、并且认真处理过",过度包装会适得其反,一旦被识破,连带殺伤论文的
   整体可信度。

---

*上一篇:[04-sentence-level-academic-english.md](04-sentence-level-academic-english.md)。下一篇:
[06-revision-methodology-and-ai-boundary.md](06-revision-methodology-and-ai-boundary.md)——多轮
修改方法论与 AI 辅助边界。*
