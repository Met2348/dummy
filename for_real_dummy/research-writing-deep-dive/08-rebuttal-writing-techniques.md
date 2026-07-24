# 08 · Rebuttal 写作技巧(Rebuttal Writing Techniques)

> 总览见 [00-roadmap.md](00-roadmap.md)。这是导师第二轮反馈里被明确点名强调的一项。07 类讲的是
> "审稿人会挑出什么样的刺",这一篇讲"刺被挑出来之后,怎么在几百字的篇幅里,尽量说服一个可能只读
> rebuttal、不会重读全部 review 的 Area Chair"。09 类 capstone 会把这一篇的技巧应用到一个完整的
> 模拟场景里。

---

## 1. 真实数据开篇:Rebuttal 到底值不值得认真写

先摆一组真实数据,回答"认真写 rebuttal 到底值不值"这个最基础的问题。一项对 ICLR 2024-2025 超过
19000 篇论文、74000 条评审的大规模分析(*Insights from the ICLR Peer Review and Rebuttal
Process*)发现:**分数在 rebuttal 后提升的论文,最终录用率是 55.7%-57.6%;分数没有变化的论文,
录用率只有 7.8%-12.4%**——差距接近五到七倍。同一项研究还发现:大多数论文的分数在 rebuttal 后
确实没有变化(2024 年 81%、2025 年 75%),分数提升的比例在增长(2024 年 17%、2025 年 23%),分数
下降的比例很低且稳定(两年都是 1%)。**时机也有讲究**:在 rebuttal 期**中段**提交的回复成功率
最高(接近三分之一带来分数提升),过早提交(可能显得仓促)和最后一刻才提交(可能是在应对已经无法
挽回的硬伤)效果都更差。

**这组数据传达的核心信息很直接**:rebuttal 不是走过场的礼貌性回复,是**投稿流程里投入产出比最高
的几个动作之一**——但它也不是万能的,分数不变的论文录用率依然远低于分数提升的论文,说明"写了
rebuttal"和"写了一份真正有说服力的 rebuttal"完全是两回事,后面几个知识点讲的就是这个"有说服力"
具体怎么做到。

**可操作检查清单:**
- [ ] 是否把 rebuttal 阶段当作和写初稿同等重要的工作来对待,预留了充分时间,而不是应付性地回复
- [ ] 提交时机是否安排在 rebuttal 期中段,而不是收到评审后立刻仓促回复,也不是拖到截止前最后
  一刻才交
- [ ] 是否意识到"分数没有变化"本身就是一个和"分数下降"性质不同、但同样值得警惕的结果——录用率
  的巨大落差说明"没有说服审稿人"这件事的代价被很多作者低估了

**量化验证:** 这一条本身是**真实统计数据的引用**,不是可以在 `.venv` 里现场跑出来的东西——如实
标注这条知识点的"验证"就是标注数据来源和样本规模,不硬造一个假的可运行例子来"证明"这组数据。
后面几个知识点会提供真正可运行的辅助工具。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"55.7%-57.6% 和 7.8%-12.4% 这两个区间为什么是个范围,不是一个精确数字?"
  → 反驳:范围来自 2024 和 2025 两年数据的差异(不同年份的具体比例略有不同),用区间而不是单一
  数字如实反映了这一点,不应该为了讲述方便就武断地只引用其中一年的数字当作"标准答案"。
- **决策依据追问轴**:"这个相关性说明的是'好的 rebuttal 导致分数提升',还是'论文本身质量高,既
  容易写出好 rebuttal,分数也容易提升'这种共同因素导致的虚假相关?" → 反驳:诚实的答案是这项统计
  本身是相关性分析,不是因果推断实验,不能排除"论文质量本身"这个共同原因的可能性——但这不改变
  对个体作者的实践含义:不管因果关系具体怎样,一份认真、有实质内容的 rebuttal 依然是作者力所能及
  范围内、性价比最高的投入之一。

**常见坑:**
1. 因为"大多数论文分数不会变"就认为"写不写都一样"——这个推理反过来看才对:正是因为大多数论文
   分数不变,能让分数提升的论文才显得格外突出,录用率差距才会这么大。
2. 把这组数据当作"只要认真写 rebuttal 就能翻盘"的保证——数据说明的是"分数提升"和"录用率"的
   关联,不是"写 rebuttal"和"分数提升"的关联,真正决定分数会不会提升的是 rebuttal 的实质内容
   质量(知识点 2-5 讲的技巧),不是"写了"这个动作本身。

---

## 2. 结构:开头总结改动 + 按主题合并回复 + 提及每位审稿人

多份公开的 rebuttal 写作指南在结构建议上高度一致:**开头用几句话总结这次回复/修订的主要变化**,
让审稿人(以及可能只读 rebuttal、不重读全部 review 的 Area Chair)能在几秒内知道"作者做出了哪些
实质性回应";正文**按主题/话题组织,而不是按审稿人逐一分段回复**——顶会审稿人一人可能要看 20-25
篇论文的 review 和 rebuttal,按主题组织能让"多位审稿人共同关心的问题"被一次性、集中地回答,而不是
同样的内容在不同审稿人的回复里重复写三遍,浪费本就紧张的字数预算;同时要确保**每位审稿人至少被
点名提及一次**,并且明确回应他们提出的至少一个具体问题,让每个人都能感觉到自己的意见被认真对待。

**常见误区/反面例子:** 严格按"Response to Reviewer 1"/"Response to Reviewer 2"/"Response to
Reviewer 3"三段分别回复,如果三位审稿人都问到了"为什么没有和 AVIC 比较",这个回答要在三段里
各写一遍——不仅浪费字数,读起来也让人觉得没有站在全局角度组织回复。

**逐处修改对照:** 把"AVIC 对比缺失"提炼成一个独立主题,合并成一段集中回答,段首注明这是哪几位
审稿人共同提出的关切("Reviewers 1 and 3 both asked about comparison to AVIC..."),既节省篇幅,
也向审稿人展示了作者认真梳理过全部意见、找出了共性,而不是逐条机械应付。

**可操作检查清单:**
- [ ] 开头是否有一段简短的"改动总结",而不是直接一头扎进逐条回复
- [ ] 是否先梳理过全部审稿人的意见,识别出哪些是多人共同提出的关切,再决定组织结构——这一步应该
  在动笔写正文之前完成,不是写的过程中临时发现"好像有点像"
- [ ] 每位审稿人是否至少被点名提及一次,并有实质性回应,不能让某一位审稿人的意见完全消失在按
  主题合并的正文里
- [ ] 是否避免了"final version"这类措辞(暗示论文已经确定被接收),改用"revised version"这种
  更谨慎的说法

**量化验证:** "怎么合并"需要判断力,但"哪些话题是多位审稿人共同提出的"是可以精确统计的——下面的
工具接收每位审稿人提出的问题列表,自动识别共性话题,给出"该合并回复"还是"该单独回复"的建议:

```python
from collections import Counter

reviewer_issues = {
    "Reviewer 1": ["missing_baseline_AVIC", "no_seed_variance", "clarity_fig2"],
    "Reviewer 2": ["no_seed_variance", "synthetic_env_only"],
    "Reviewer 3": ["missing_baseline_AVIC", "synthetic_env_only", "voc_novelty_question"],
}

def plan_rebuttal_budget(reviewer_issues, total_words=750):
    all_issues = [iss for issues in reviewer_issues.values() for iss in issues]
    counts = Counter(all_issues)
    shared = {k: v for k, v in counts.items() if v > 1}
    unique = {k: v for k, v in counts.items() if v == 1}
    n_topics = len(counts)
    per_topic = total_words / n_topics
    return {
        "n_reviewers": len(reviewer_issues),
        "n_distinct_topics": n_topics,
        "shared_topics(merge into one reply)": shared,
        "unique_topics(reply separately)": list(unique.keys()),
        "words_per_topic_if_even": round(per_topic, 1),
    }

plan = plan_rebuttal_budget(reviewer_issues)
assert plan["shared_topics(merge into one reply)"] == {"missing_baseline_AVIC": 2, "no_seed_variance": 2, "synthetic_env_only": 2}
assert plan["n_distinct_topics"] == 5
for k, v in plan.items():
    print(f"{k}: {v}")
print("OK: 7 comments from 3 reviewers dedupe to 5 real topics, 3 of which are shared concerns -- merge into one reply instead of writing 3 separate ones")
```

本机实测:3 位审稿人一共提出 7 条(带重复计数)意见,去重后是 5 个真实不同的话题,其中 3 个
(missing_baseline_AVIC / no_seed_variance / synthetic_env_only)被至少两位审稿人共同提到,应该
合并成 3 段集中回复,而不是拆成 7 段各自独立的回复。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"按主题合并会不会让某位审稿人觉得'我的意见被淹没在别人的意见里,没有得到
  单独重视'?" → 反驳:这正是为什么结构要求"每位审稿人至少被点名提及一次"——合并回复的正文里,
  应该明确写出"Reviewers 1 and 3 both raised this concern",让每位审稿人都能在文本里找到自己
  被提及的位置,合并的是篇幅,不是身份认同。

**常见坑:**
1. 合并回复时把不同审稿人问题的细微差异抹平了——比如两位审稿人都问"为什么没对比 AVIC",但一位
   问的是"性能对比",另一位问的是"方法论差异",合并回复必须同时覆盖两个角度,不能用同一句笼统
   的话糊弄过去。
2. 只统计了话题出现次数,没有考虑话题的严重程度——知识点 5 会展开这一点:话题出现的频率和话题
   对最终分数的影响程度是两个独立的维度,篇幅分配不能只看频率。

---

## 3. 语言技巧:承认是"weakness"不是"flaw",是"revised version"不是"final version"

多份公开的 rebuttal 实战笔记(包括长期从事同行评审的研究者的个人经验总结)都提到一套具体的措辞
习惯:**避免"flaw"/"mistake"/"unoriginal"/"incremental"这类几乎等于自我定罪的词,换成
"weakness"/"issue"/"limited"/"extending existing work"这类留有讨论空间的措辞**;不说"final
version"(暗示论文已经确定被接收,容易显得托大),说"revised version"。这不是粉饰太平,而是把
语言的"预设立场"调整成中性——"flaw"这个词本身已经预设了"这是个不该出现的错误",而很多审稿人
指出的问题其实是"设计选择的局限",不是"错误",用词的预设立场如果比实际情况更严重,反而会不必要
地放大问题的严重程度。

**常见误区/反面例子:** "We acknowledge this is a flaw in our experimental design and a mistake
in how we framed the contribution."——用了"flaw"和"mistake"两个高预设立场的词,把一个可以讨论
的局限,写成了近乎自我认罪的陈述。

**逐处修改对照:** "We agree this is a limitation of our current experimental scope, and we
have revised the framing of this contribution accordingly in the revised version."——用
"limitation"和"revised"替换掉"flaw"和"mistake"/"final",态度同样诚恳,但没有预设"这是个不该
发生的错误"。

**可操作检查清单:**
- [ ] 通读一遍 rebuttal 草稿,查找"flaw"/"mistake"/"unoriginal"/"incremental"这类高预设立场的
  词,评估是否可以替换成"weakness"/"issue"/"limited"/"extending existing work"而不改变实质意思
- [ ] 检查是否出现"final version"这个说法,统一替换成"revised version"
  (来源:多位研究者公开的 rebuttal 写作笔记,这条建议高度一致)
- [ ] 语言整体是否保持"powerful and positive"的基调——不是回避问题,而是暗示"这些点只需要小幅
  澄清/修订就能解决",而不是暗示"存在动摇整体结论的严重问题"(前提是这个暗示本身要诚实,不能对
  真正严重的问题也用这种轻描淡写的语气,那会构成过度粉饰,见知识点 4 的判断标准)
- [ ] 是否在感谢审稿人和承认好的建议之间保持了分寸——感谢是应有的礼貌,但不宜过度,一两句得体的
  感谢加上认可具体的好建议,比大段客套话更有效

**量化验证:** "这个措辞是不是预设立场过重"是语感判断力问题,但"是否使用了这份已知的高风险词表"
是可以用简单的字符串检索精确统计的——这个检查逻辑足够简单,不需要单独写一段代码,直接给出规则:
在提交前对 rebuttal 全文做一次检索,搜索 "flaw"、"mistake"(在承认己方问题的语境下)、
"unoriginal"、"final version" 这几个词,每处命中都值得重新审视措辞,这条规则复用
[04 类](04-sentence-level-academic-english.md)套话检测器的思路(维护一份关键词清单、统计命中),
这里不重复贴代码。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"换个词就能改变审稿人对问题严重性的判断吗?这是不是纯粹的话术操纵?" → 反驳:
  措辞调整不能改变问题的客观严重程度,但能避免**不必要地**放大审稿人对严重程度的主观感受——如果
  问题本身确实是"flaw"级别(比如结果不可复现、存在明显的实验设计错误),诚实用"flaw"这个词才是
  该做的事,这条技巧的适用范围是"审稿人指出的是真实存在但没有那么致命的局限",不是教人把真正
  严重的问题也粉饰成无关紧要,这条边界必须守住,详见知识点 4。

**常见坑:**
1. 把这套措辞技巧用在真正严重的问题上——如果审稿人指出的确实是会动摇核心结论的硬伤,用轻描淡写
   的语气回应会显得不诚恳,甚至可能让审稿人觉得作者没有真正理解问题的严重性,这比用词本身更伤
   可信度。
2. 只改了表层用词,没有改变论证的实质内容——措辞调整是最后一层打磨,不能替代把问题真正想清楚、
   给出实质性回应这个更重要的工作。

---

## 4. "数据不够/缺少实验"类质疑:能做就承诺具体的,不能做就论证为什么不是决定性的

审稿人最常见的质疑类型之一是"数据/实验不够充分"(缺基线、缺消融、缺场景覆盖)。这类质疑的应对
方式应该取决于一个具体的判断:**这个补充实验能不能在 rebuttal 窗口内真正做出来**。能做到,就
承诺得非常具体(哪怕只是初步结果),而不是空泛地说"我们会补充这个实验";做不到,就需要论证"这个
缺口是否真的动摇了核心结论",如果确实动摇,诚实的应对是承认这是一个范围边界(呼应
[05 类](05-limitations-and-honest-disclosure.md)),而不是硬辩。

**常见误区/反面例子:** 对"你们没有和 AVIC 直接对比"这类质疑,笼统回复"We will add this comparison
in the revised version"——审稿人读完不知道这个承诺有多可信(是已经在做了,还是随口一说),也不
知道大概会得到什么结果。

**逐处修改对照:** 如果能在 rebuttal 窗口内真正跑出初步结果:"We agree and have run a preliminary
comparison against AVIC-R on our synthetic benchmark: our method reaches a 82.0% hit rate vs.
AVIC-R's approximately X%. We will include the full comparison with error bars in the revised
version."——给出了一个具体的、哪怕是初步的数字,比空泛的承诺可信得多。

**可操作检查清单:**
- [ ] 对每一条"数据不够"类质疑,是否先判断了"这个补充实验在 rebuttal 窗口内是否真的可行",而不是
  不加区分地一律承诺"我们会补充"
- [ ] 可行的情况下,是否尽量给出哪怕是初步/部分的具体数字,而不是空泛的承诺
- [ ] 不可行的情况下,是否论证了"这个缺口不改变核心结论"的具体理由,而不是简单地说"没有时间做"
  (这类理由容易被解读为敷衍)
- [ ] 如果这个缺口确实可能动摇核心结论,是否诚实承认这是一个范围边界,而不是继续辩护

**量化验证:** "这个补充实验可不可行"和"这个缺口是否动摇核心结论"都需要研究者自己的专业判断,
不可 assert;但**给定这两个判断的结果之后,该选哪种回应策略**是可以用一个决策规则明确表达的:

```python
def choose_response_strategy(request, feasible_within_rebuttal_window, changes_core_claim):
    if feasible_within_rebuttal_window:
        return {
            "strategy": "commit_with_concrete_plan",
            "template": f"We agree and will add {request}. Preliminary result: <concrete number>. "
                        f"Full result will appear in the revised version.",
        }
    if not changes_core_claim:
        return {
            "strategy": "argue_not_decisive",
            "template": f"We could not run {request} within the rebuttal window. We argue this "
                        f"does not undermine our core claim because <specific reason>, and we "
                        f"will discuss it explicitly as a limitation in the revised version.",
        }
    return {
        "strategy": "concede_scope_limit",
        "template": f"We acknowledge {request} is outside the scope we can validate here; we "
                    f"revise our claims to explicitly state this boundary.",
    }

case_feasible = choose_response_strategy("5 more random seeds", True, changes_core_claim=False)
case_infeasible_minor = choose_response_strategy("a qualitative case study", False, changes_core_claim=False)
case_infeasible_major = choose_response_strategy("real-robot validation", False, changes_core_claim=True)

assert case_feasible["strategy"] == "commit_with_concrete_plan"
assert case_infeasible_minor["strategy"] == "argue_not_decisive"
assert case_infeasible_major["strategy"] == "concede_scope_limit"
for name, case in [("feasible within rebuttal window", case_feasible),
                    ("infeasible but does not affect core claim", case_infeasible_minor),
                    ("infeasible and affects core claim", case_infeasible_major)]:
    print(f"{name}: strategy={case['strategy']}")
```

本机实测:三种场景精确对应三种不同的回应策略。**明确边界**:这个函数只是把"决策逻辑"显式化,
`feasible_within_rebuttal_window` 和 `changes_core_claim` 这两个布尔值本身仍然需要研究者基于
真实情况诚实判断,工具不能替代这两个判断,只能保证"判断做出之后,选对应策略这一步不会出错"。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"如果'做不到且影响核心结论'这种情况发生在 rebuttal 阶段才被发现,是不是
  已经太晚了?" → 反驳:确实太晚了——这正是为什么 07 类讲的"提前自查红旗"如此重要,rebuttal 阶段
  才第一次意识到某个缺口会动摇核心结论,说明投稿前的自我审查(呼应 [06 类](06-revision-methodology-and-ai-boundary.md)
  知识点 5"二次核验")做得不够,rebuttal 只能亡羊补牢,不能替代投稿前就该完成的工作。

**常见坑:**
1. 对"不可行"的判断过于草率,没有真正评估清楚补充实验需要多少时间/资源就直接放弃——rebuttal
   窗口通常有一到两周,一些规模不大的补充实验(比如多跑几个随机种子)其实是可行的,不要因为
   "感觉来不及"就不去尝试。
2. 承诺了"会在 revised version 里补充",但最终提交的修订版真的没有兑现——这种承诺一旦写进
   rebuttal,就应该被当作对审稿人的正式承诺认真兑现,不能只是缓兵之计。

---

## 5. 字数限制内的取舍:篇幅要向真正可能扭转分数的问题倾斜

多份 rebuttal 指南强调字数/字符限制的重要性——很多会议明确要求 rebuttal 控制在 500-750 词左右,
超长的 rebuttal 即使写得很认真也可能因为超限被直接无视。在这个有限预算内,一个容易被忽视的判断是:
**不应该不分轻重地平均分配篇幅**。知识点 2 的代码演示了"哪些话题该合并"，这里更进一步:同样是需要
回复的话题,对最终分数的潜在影响程度并不相同(比如"没有和最重要的一篇最近邻工作对比"通常比"某个
图的坐标轴标签不够清楚"更可能真正扭转审稿人的判断),篇幅分配应该向前者倾斜。

**常见误区/反面例子:** 把有限的 750 词,平均分给识别出的 5 个话题,每个话题正好 150 词——不管
这个话题是"核心方法论质疑"还是"一个图注的小笔误"。

**逐处修改对照:** 给每个话题一个粗略的"严重程度"估计(这条意见有多大概率真的影响最终分数),
把预留给"保底覆盖"之外的剩余篇幅,按严重程度加权分配,而不是平均分配。

**可操作检查清单:**
- [ ] 是否给每个需要回复的话题都有一个大致的"重要程度"判断(哪怕只是主观的高/中/低三档),而不是
  默认所有话题同等重要
- [ ] 篇幅分配是否明显向"可能真正扭转分数"的话题倾斜,次要问题(比如图注笔误)是否只用简短的
  一两句话确认修正,不占用过多篇幅
- [ ] 每个话题即使分到的篇幅很少,是否依然至少有一句实质回应,而不是完全略过——略过比回应过短
  更容易被解读为回避

**量化验证:**

```python
# 每条意见的"severity"是作者对"这条意见有多大概率真的影响最终分数"的主观估计(1-5),
# 不是审稿人给出的客观数字——分配预算时把有限字数向severity更高的问题倾斜,
# 而不是不管轻重平均分配
concerns = [
    {"topic": "missing_baseline_AVIC", "severity": 5},
    {"topic": "no_seed_variance", "severity": 4},
    {"topic": "synthetic_env_only", "severity": 3},
    {"topic": "clarity_fig2", "severity": 1},
    {"topic": "voc_novelty_question", "severity": 5},
]

def weighted_budget(concerns, total_words=750, min_words_per_topic=40):
    total_severity = sum(c["severity"] for c in concerns)
    remaining = total_words - min_words_per_topic * len(concerns)
    assert remaining >= 0, "total budget cannot cover the minimum words per topic"
    allocation = []
    for c in concerns:
        extra = remaining * (c["severity"] / total_severity)
        allocation.append({"topic": c["topic"], "words": round(min_words_per_topic + extra)})
    return allocation

weighted = weighted_budget(concerns)
top_topic = max(weighted, key=lambda a: a["words"])
bottom_topic = min(weighted, key=lambda a: a["words"])
assert top_topic["topic"] in ("missing_baseline_AVIC", "voc_novelty_question")
assert bottom_topic["topic"] == "clarity_fig2"
assert top_topic["words"] > bottom_topic["words"]
print(f"even split (severity-blind): about {750 // len(concerns)} words each")
print("severity-weighted allocation:")
for a in weighted:
    print(f"  {a['topic']}: {a['words']} words")
```

本机实测:平均分配下每个话题约 150 词;按严重程度加权后,最高优先级的两个话题(missing_baseline_AVIC
和 voc_novelty_question)各分到 193 词,最低优先级的 clarity_fig2 只分到 71 词(依然保留了
`min_words_per_topic=40` 之上的一点余量,不是完全略过)。**明确边界**:`severity` 这个数字本身
是作者主观估计,不是客观测量——这个工具能做的是"给定你自己对轻重的判断之后,把篇幅精确地分配下去",
不能替你判断"这条意见到底有多重要",那依然需要基于对审稿人真实意图和论文实际情况的理解。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"severity 的估计如果错了(比如低估了一个看起来次要、实际上审稿人非常在意的
  问题),这个加权分配反而会帮倒忙?" → 反驳:确实存在这个风险——严重程度估计本身就是一个需要
  仔细读懂审稿人真实关切的判断,如果拿不准,更保守的做法是给可能被低估的话题留出比工具建议更多
  一点的余量,工具的输出应该被当作一个起点而不是最终定论,人工复核依然是必要的最后一步。

**常见坑:**
1. 把"重要的问题"等同于"审稿人语气更激烈的问题"——审稿人的语气强弱和问题客观上是否致命是两个
   不完全相关的维度,判断严重程度应该基于问题本身对核心结论的影响,不是单纯看审稿人的措辞情绪。
2. 为了给"重要"话题挤出更多篇幅,把次要话题完全略去不提——即使分到的篇幅很少,略去不回应和
   简短回应给读者的印象完全不同,前者容易被解读为回避,后者只是合理的篇幅取舍。

---

*上一篇:[07-reviewer-perspective-and-rejection-patterns.md](07-reviewer-perspective-and-rejection-patterns.md)。
下一篇:[09-mock-review-rebuttal-capstone.md](09-mock-review-rebuttal-capstone.md)——收尾 capstone,
模拟审稿意见 + rebuttal 攻防。*
