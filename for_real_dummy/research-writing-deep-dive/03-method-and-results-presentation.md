# 03 · Method/实验部分的呈现逻辑(Method & Results Presentation)

> 总览见 [00-roadmap.md](00-roadmap.md)。这一篇讲的是"怎么让审稿人 5 分钟内 get 到核心贡献"——审稿人
> 一年要读几十上百篇同方向论文,Method/实验部分的呈现逻辑决定了他们是"迅速抓住重点"还是"迷失在细节里"。

---

## 1. 结果先行 vs 方法先行:写作顺序和阅读顺序是两件事

一个容易被忽略的事实:**论文不需要按最终阅读顺序撰写**。多个真实的写作指南建议反过来操作——先把
结果表/图填进去(哪怕数字还没跑完整,先占位),再倒推着写"这组结果需要哪些实现细节支撑",最后才组织
成读者看到的正向顺序(先方法后结果)。这是"写作顺序"的建议,不要和"最终呈现给读者的顺序"混淆——
最终呈现给读者时,"结果先行"还是"方法先行"是另一个独立的判断:如果方法本身没有复杂到需要单独建立
概念才能理解结果,可以让结果更早出现、增强说服力;如果方法有一个不先解释就看不懂结果的核心机制,
必须先讲方法。

**常见误区/反面例子:** 把"写作顺序"和"呈现顺序"混为一谈,导致两种常见失败:①严格按 Introduction
→ Method → Results 的顺序**撰写**,方法部分写到一半发现"这里其实需要一个例子",但因为还没写到
Results,只能用假想的抽象描述,写完 Results 才发现方法部分那个例子的假设和实际结果对不上,只能返工;
②在**呈现**顺序上,把所有实现细节(超参数、训练技巧、工程优化)一股脑放在结果之前,读者要读完两三页
"我们用了 Adam,学习率 3e-4,batch size 128……"才能看到"所以效果怎么样"。

**逐处修改对照:**

| 问题模式 | 后果 | 改法 |
|---|---|---|
| 撰写顺序按 Intro→Method→Results 严格线性推进 | 方法部分写到需要具体例子时无据可依,容易写得抽象 | 先把
  结果表/图填进去(哪怕数字占位),倒推着确定方法部分到底需要交代哪些细节 |
| 呈现顺序把全部实现细节堆在结果之前 | 读者被工程细节淹没,读到"效果怎么样"时已经失去耐心 | 只保留
  "理解结果所必需"的方法描述在结果之前,其余训练技巧类细节挪到附录或"实现细节"小节 |

**可操作检查清单:**
- [ ] 撰写阶段是否从"最有把握、最想讲的实验结果"开始动笔,而不是死板地从 Introduction 第一句开始
  往后写——先写自己最熟悉的部分,回头组织顺序时更容易
- [ ] 呈现给读者的方法描述,是否只保留"理解后续结果所必需"的内容,工程细节(超参数网格、具体的库
  版本号)能不能挪到附录
- [ ] 每个 Section/Subsection 的开头一句和结尾一句,是否能单独拎出来看懂"这一节在全文叙事里的作用"
  ——写作过程中先规划好每段的开头句和结尾句,再填充中间内容,是让论文读起来"叙事紧凑"而不是
  "松散罗列"的一个具体可执行技巧

**量化验证:** "撰写顺序该怎么安排"是纯工作习惯,不可 assert;但呈现给读者的顺序里"方法细节堆积在
结果之前的比重"是可以粗略量化的——用一个简化代理:统计"结果部分第一次出现具体数字"之前,方法部分
用了多少词,这个数字越大,读者要读的"前戏"就越长。这个检查本身逻辑很简单,不需要专门写代码验证,
在这里给出规则即可:如果方法部分超过全文正文的 40%-50%,同时结果部分的第一个具体数字要等到方法部分
完全结束才出现,这是一个值得警惕的信号——真正需要验证的是"方法部分是否只保留了必要内容",这件事
留给知识点 3(消融/呈现规范)和 [02 类](02-introduction-and-related-work.md)知识点 5 的 claims-evidence
检查器一起做,这里不重复造轮子。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"你选择结果先行,是真的判断过读者需求,还是单纯为了'吸引眼球'?" → 反驳:
  判断依据应该是"方法本身是否有一个不先讲就看不懂结果的核心机制"——如果方法只是"标准做法的组合",
  结果先行没有理解障碍;如果方法引入了一个新概念(比如一个此前没定义过的量,如"信息渗透率"),必须
  先建立这个概念,结果先行反而会让读者看到数字却不知道在测什么。
- **方案批判迭代轴**:"你把大量超参数细节挪到附录,会不会被质疑'为了让正文好看,牺牲了可复现性'?" →
  反驳:可复现性和呈现顺序是两个独立的要求——附录/补充材料本身就是复现细节该待的地方,只要正文里
  有清晰的指引("完整超参数见 Appendix B"),挪动位置不等于隐藏信息,这条边界在
  [paper-submission-deep-dive/05 号文件第 3 节](../paper-submission-deep-dive/05-camera-ready-and-supplementary.md#3-supplementary-material-怎么组织--不喧宾夺主的原则)进一步展开。

**常见坑:**
1. 把"结果先行"理解成"结果部分要先呈现,所以先写方法但后写的时候强行倒着排版"——顺序调整应该在
   规划阶段就想清楚,而不是写完之后简单地把段落顺序拖拽调换,拖拽调换出来的文字过渡通常很生硬。
2. 反过来,方法部分写得过于详尽以至于喧宾夺主,超过实际需要的篇幅,却没有意识到这是可以精简的——
   判断标准始终是"理解后续结果是否必需",不是"我做了这些工作所以都要写出来"。

---

## 2. Figure 1 该放什么:结论性证据,不是流程示意图

多份独立的审稿人自述和"如何读论文"方法论一致指向同一个阅读模式:**摘要 → 图(尤其是第一张图)→
结论,全程只花 5-10 分钟**。这意味着 Figure 1 的角色极其关键——如果 Figure 1 只是一张"系统架构/
流程示意图"(展示模块 A 连到模块 B 连到模块 C),读者看完知道"这个系统长什么样",但不知道"这个系统
到底行不行"。更有效的 Figure 1 是**直接呈现最有说服力的量化对比**(哪怕是简化版本),让读者在看完
第一张图后就已经形成"这篇论文大概率有干货"的第一印象。

**常见误区/反面例子:** Figure 1 是一张纯示意图,标题为"Overview of our controller architecture and
its three-stage pipeline"——图和标题都没有传达任何"效果如何"的信息,读者需要翻到实验部分才能看到
第一个数字。

**逐处修改对照:** 把 Figure 1 换成一张直接呈现关键对比的图,标题相应改成量化陈述——"Task-conditioned
imagination reaches 82.0% hit rate vs. 63.7% for unconditioned imagination across three target
settings"。如果确实需要一张架构图帮助理解方法,把它降级为 Figure 2 或者放进 Method 一节内部,不占据
读者最先看到、也是记忆最深的 Figure 1 位置。

**可操作检查清单:**
- [ ] Figure 1 的标题读起来像不像一个可以脱离全文单独成立的结论——如果把标题单独摘出来发在社交媒体
  上,别人能不能看懂"这篇论文大概做出了什么"
- [ ] Figure 1 是否包含至少一个量化对比,而不是纯粹的模块连线图
- [ ] 如果确实需要架构示意图,是否把它安排在 Figure 1 之后的位置,而不是让它占据读者最先看到的位置
- [ ] Figure 1 的标题和摘要的关键数字是否一致——两者应该互相印证,不能一个说 82.0%、另一个说别的数字

**量化验证:** "这张图是否真的有说服力"是判断力问题,但"标题读起来更像流程图描述还是更像量化结论"
是可以用关键词代理粗略区分的:

```python
import re

PROCESS_WORDS = ["overview of", "pipeline", "illustration of", "architecture of", "diagram of",
                  "flowchart", "schematic"]
EVIDENCE_WORDS = ["compared to", "vs.", "outperforms", "higher than", "lower than"]
# 用带小数点的数字或百分号作为"量化结果"信号,不用裸\d——第一版用裸\d时被
# "Figure 1:"这个图号前缀本身的数字"1"污染,process_caption被误判成有量化结果,
# 已修复为只认百分比/小数这类"看起来像指标"的数字模式
RESULT_NUMBER_RE = re.compile(r"\d+(\.\d+)?\s?%|\d+\.\d+")

def classify_figure1_caption(caption):
    lower = caption.lower()
    is_process = any(w in lower for w in PROCESS_WORDS)
    has_result_number = bool(RESULT_NUMBER_RE.search(caption))
    is_evidence = any(w in lower for w in EVIDENCE_WORDS) or has_result_number
    return {"looks_like_process_diagram": is_process, "looks_like_evidence_figure": is_evidence}

process_caption = "Figure 1: Overview of our controller architecture and its three-stage pipeline."
evidence_caption = ("Figure 1: Task-conditioned imagination reaches 82.0% hit rate vs. 63.7% for "
                     "unconditioned imagination across three target settings.")

r1 = classify_figure1_caption(process_caption)
r2 = classify_figure1_caption(evidence_caption)
assert r1["looks_like_process_diagram"] and not r1["looks_like_evidence_figure"]
assert r2["looks_like_evidence_figure"]
print("process_caption (bad example, Figure 1 is just a process diagram):", r1)
print("evidence_caption (revised, Figure 1 is direct evidence):", r2)
```

本机实测:`process_caption` 命中 `looks_like_process_diagram=True`、`looks_like_evidence_figure=False`;
`evidence_caption` 相反。**真实撞到的坑**:第一版 `RESULT_NUMBER_RE` 直接用裸 `\d`(任意数字)判断
"是否包含量化结果",结果 `process_caption` 被误判成"有量化结果"——因为标题前缀"Figure 1:"本身就带了
一个数字"1",把图号编号污染成了"检测到数字"。这和
[04 类](04-sentence-level-academic-english.md)"引用是脚注不是名词"知识点里 `et al.` 污染句子切分
是同一类教训:**写文本分析代码时,不能假设输入是"干净"的,格式性质的数字/标点(图号、章节号、
参考文献编号)很容易被朴素的正则规则误当成实质内容**——已改用"带小数点或百分号"这种更贴近"真实指标"
形状的模式修复。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"Figure 1 里的 82.0% vs 63.7%,和摘要里的数字是不是同一组实验、同一个设置?"
  → 反驳:必须是同一组——如果 Figure 1 为了视觉效果换了一个更好看但和摘要不同的子集数字,这是会被
  审稿人当场抓到的不一致,比"Figure 1 不够吸引人"严重得多。

**常见坑:**
1. Figure 1 塞入过多维度(既要看趋势又要看对比又要看消融),信息密度过高反而丧失了"5 分钟看懂"的
   初衷——Figure 1 应该只服务一个最核心的论点,其余维度放进后续图表。
2. 图注(caption)只写"如图所示"没有独立说明——好的图注应该能脱离正文单独被理解(这一点和
   [research-figures-deep-dive/05 号文件第 1 节"自洽性原则"](../research-figures-deep-dive/05-caption-writing.md#1-自洽性原则caption-要能脱离正文被理解)直接相关)。

---

## 3. 消融研究:受控基线 + 预先设定的清单 + 统计显著性

消融研究(ablation study)是"证明每个组件真的有必要"最直接的方式,但一个常见的失败模式是把消融做成
"试错式调参"——改一改看看结果变没变,而不是**提前确定一个固定的基线配置和一份要拆解的组件清单**,
逐项拆、逐项测,并且报告是否有统计显著性。缺失消融是审稿人最常见的抱怨之一,直接后果是读者/审稿人
没法判断"论文声称的提升,到底来自哪个具体设计,还是几个改动叠加的整体效果、其中可能只有一个真正
起作用"。

**真实案例(AVIC 论文 Table 6,`research/world-model-imagination-controller/01-meeting-briefing.md`
已核验引用):** AVIC 的训练目标里有一项"wrong-skip 惩罚"(错误地跳过想象要扣分,防止策略摆烂全部
skip)。消融实验去掉这一项之后,准确率从 77.33% 掉到 62.67%——掉了 14.66 个百分点,这是四项训练目标
里唯一让论文作者专门在正文强调"这是训出可用策略的关键"的一项。这正是消融研究该有的呈现方式:**受控
(只改一个变量)、量化(给出精确的 delta)、有解释力(读者一眼看出"原来这个组件才是关键,不是别的")**。

**可操作检查清单:**
- [ ] 消融的基线配置是否固定不变、每次只拆一个组件——如果同时改动多个组件,结果无法归因到具体是
  哪个改动起了作用
- [ ] 消融列表是否在实验设计阶段就确定,而不是看到结果后"挑几个看起来有戏的方向"回填——事后挑选
  容易selection bias,读者也无法分辨这份清单是不是精心挑选过的
- [ ] 每组消融结果是否报告了和完整模型的精确 delta,而不是只给绝对数字让读者自己心算差值
- [ ] 结果是否附带方差/显著性说明(呼应 [07 类](07-reviewer-perspective-and-rejection-patterns.md)
  会展开的"无种子方差"红旗)——一个消融的 delta 如果落在噪声范围内,不能被当作"这个组件很重要"的
  证据

**量化验证:**

```python
# 真实数字来自 AVIC 论文(Yu et al., arXiv:2602.08236)Table 6 消融实验,
# world-model项目01-meeting-briefing.md已核验并引用过这组数字
ablation_rows = [
    {"config": "full (with wrong-skip penalty)", "accuracy": 77.33},
    {"config": "w/o wrong-skip penalty", "accuracy": 62.67},
]

def ablation_deltas(rows, full_key="full (with wrong-skip penalty)"):
    full_acc = next(r["accuracy"] for r in rows if r["config"] == full_key)
    out = []
    for r in rows:
        if r["config"] == full_key:
            continue
        delta = r["accuracy"] - full_acc
        out.append({"removed_component": r["config"], "delta_pp": round(delta, 2)})
    return out

deltas = ablation_deltas(ablation_rows)
assert deltas[0]["delta_pp"] == -14.66
print("ablation results (change vs full model, in percentage points):", deltas)
print(f"OK: removing the wrong-skip penalty drops accuracy by {abs(deltas[0]['delta_pp'])} pp -- "
      f"this is what presenting 'which component matters' should look like: controlled baseline + explicit delta, not a bare claim")
```

本机实测:`delta_pp = -14.66`,和 AVIC 原文一致。这个函数本身很朴素(就是减法),但它示范的是**呈现
规范**而不是计算难度——很多论文的消融表只给绝对数字,让读者自己心算差值,规范的做法是直接算好 delta
摆出来,降低读者的认知负担。

**审稿人会怎么挑刺 + 反驳链:**
- **方案批判迭代轴**:"这个消融只做了'去掉整个 wrong-skip 惩罚项',能不能进一步拆分——比如只降低
  惩罚权重而不是完全去掉,才能说清楚是'需要这个机制'还是'需要这么强的权重'?" → 反驳:如果审稿人的
  追问成立,说明当前的消融粒度确实不够细——诚实的应对是承认这是"存在 vs 不存在"这个更粗粒度问题的
  答案,更细粒度的权重扫描是合理的后续要求,不是消融研究本身的方法论错误,是覆盖范围可以更完整。

**常见坑:**
1. 消融清单只包含"删除会让结果变差"的组件,不包含任何"删除后结果不变甚至变好"的组件——一份诚实的
   消融研究应该既能证明关键组件确实关键,也不回避"某个复杂设计其实没有必要"这种不那么讨喜的结果。
2. 消融实验用和主实验不同的随机种子/评测协议,导致数字之间不能直接比较——消融的可信度建立在"除了
   被拆的那个组件,其余一切完全一致"这个前提上,评测协议不一致会让整个消融失去意义。

---

## 4. Bake-off 陷阱:排名算得出来,原因算不出来

"bake-off"式实验(把自己的方法和一堆基线放在同一个表里比分数,证明自己更高)是几乎所有论文都会做的
事,但早在 ICML 的"Crafting Papers on Machine Learning"写作指南里就明确提醒过这类实验的局限:
这类研究本身几乎不能告诉我们"为什么一个方法会比另一个表现更好"的任何信息——单纯"打榜获胜"不提供
**关于原因**的任何洞察,而科学研究恰恰需要这种关于因果的洞察。换句话说:一张排名表能回答"谁的分数
更高",但回答不了"为什么更高"——后者只能靠消融研究、案例分析、误差分析这些更深入的手段。

**常见误区/反面例子:** 论文的核心实验部分只有一张排名表(方法 A/B/C/D 的分数对比),外加一句"我们的
方法显著优于所有基线",没有任何进一步的分析——审稿人读完知道"这篇论文的方法分数最高",但完全无法
判断这个提升是不是来自一个真正新颖、可迁移的洞察,还是恰好在这个数据集上蒙对了一组超参数。

**逐处修改对照:** 排名表本身不需要删除(仍然是必要的整体证据),但必须补充至少一层"为什么"——可以
是消融(知识点 3)、案例分析(挑几个方法赢/输的具体例子逐一分析)、或者误差模式分析(方法在哪类
子问题上赢得最多,这类子问题有什么共性)。

**可操作检查清单:**
- [ ] 排名表之外,是否至少有一种"为什么赢"的分析(消融/案例分析/误差模式分解)
- [ ] 如果篇幅只够放一张表,这张表本身能不能设计成"顺便回答一部分为什么"——比如按子任务类别拆分
  报告(呼应 world-model 项目里"action-conditioned 类问题 WM 带来 +57.1% 提升,但 dynamics-
  understanding 类只有 +28.5%"这种拆分粒度,比一个笼统的总分更有解释力)
- [ ] "我们的方法显著优于基线"这句话是否有具体支撑,还是排名表本身唯一的文字总结

**量化验证:** 用代码直接演示这个局限本身——排名可以精确计算,"为什么"字段永远是空的,不会因为
写更多代码而自动填上:

```python
scores = {"OurMethod": 82.0, "AVIC-R": 77.33, "GPT-4o-policy": 69.3, "RandomBaseline": 50.0}

def rank_only(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, None  # 第二个返回值故意留空:排名代码算得出,"为什么"代码算不出

ranked, insight = rank_only(scores)
assert insight is None
print("ranking (everything a bake-off can compute):", ranked)
print("why the top method wins (insight):", insight)
print("OK: the code can compute exactly who scores highest, but 'wins because of what mechanism' "
      "must come from ablations/case analysis -- it never falls out of the ranking table itself")
```

本机实测:`ranked` 精确给出四个方法从高到低的排序,`insight` 恒为 `None`——这不是代码能力不够,是
排名这类统计量在数学上就不携带"原因"这个信息,不管用什么排序算法都一样,这正是这个知识点想说明的
边界。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"你补充的案例分析,选的是随便几个例子,还是有代表性的抽样?" → 反驳:案例
  分析如果只挑"方法表现最好看"的几个例子,本身也是另一种误导(选择性呈现),更站得住的做法是说明
  选择标准(比如按误差大小排序取前后若干个,或者按子任务类别系统采样),不能是"我觉得这几个例子
  说明问题"这种没有采样规则的挑选。

**常见坑:**
1. 只在 Introduction/摘要宣称"我们的方法在多个基准上取得 SOTA",却没有配套任何解释性分析——这种写法
   在 2020 年代的审稿标准下越来越容易被明确点名要求补充。
2. 消融/案例分析写得比主实验表还长,喧宾夺主——解释性分析应该服务主结论,篇幅要有节制,不是"能写多
   细就写多细"。

---

## 5. 参数扫描类结果的呈现:用真实数据表格代替"大概是这个趋势"

很多实验天然是"扫一个参数、看结果怎么变"的形状(比如扫想象深度、扫候选数量、扫模型规模),这类结果
最容易犯的错误是只用一句文字描述趋势("随着 H 增大,决策改变率也在增大"),不给出具体数字表格——
读者没法判断这个趋势是"从 5% 涨到 6%"这种几乎可以忽略的噪声,还是真实、大幅度的变化。

**真实案例(取材 `research/world-model-imagination-controller/01-meeting-briefing.md` §3.3):** 项目组
在诊断性 pilot 里扫了想象深度 H(固定候选数 K=5),用表格直接呈现每个 H 值对应的决策改变率(5 个随机
种子均值),读者一眼就能看出这是从 0.100 稳定爬升到 0.231 的真实趋势,不是噪声——这种"表格先行,
文字总结趋势"的顺序,比反过来"先用文字描述趋势,读者需要的话自己去找表格核实"更符合"审稿人先看图表
再读文字"的真实阅读习惯(呼应知识点 2 的调研结论)。

**可操作检查清单:**
- [ ] 参数扫描类结果是否有明确的表格/图,而不是只有一句"随着 X 增大,Y 也在增大"的文字描述
- [ ] 表格是否包含方差信息(如"5 个随机种子均值±标准差"),让读者能判断趋势是否显著
  (world-model 项目原始数据本身就是"均值±标准差"格式,这不是巧合,是同一条"报告方差"纪律的体现)
- [ ] 趋势描述是否量化("从 0.100 单调升到 0.231"),而不是模糊地说"明显增大"

**量化验证:**

```python
# 真实数字来自 research/world-model-imagination-controller/01-meeting-briefing.md §3.3
# (扫想象深度H,固定候选数K=5,5个随机种子均值)
h_scan = [
    {"H": 1, "decision_change_rate": 0.100},
    {"H": 2, "decision_change_rate": 0.125},
    {"H": 3, "decision_change_rate": 0.163},
    {"H": 5, "decision_change_rate": 0.169},
    {"H": 8, "decision_change_rate": 0.231},
]

def render_table(rows, key_col, val_col):
    lines = [f"| {key_col} | {val_col} |", "|---|---|"]
    for r in rows:
        lines.append(f"| {r[key_col]} | {r['decision_change_rate']:.3f} |")
    return "\n".join(lines)

print(render_table(h_scan, "H", "decision_change_rate"))

rates = [r["decision_change_rate"] for r in h_scan]
assert rates == sorted(rates)  # 应该是单调递增趋势(H越深,决策改变率越高)
print(f"\nOK: as H goes from 1 to 8, decision-change rate rises monotonically from {rates[0]:.3f} to {rates[-1]:.3f} "
      f"-- a real data table instead of a vague 'roughly this trend' claim")
```

本机实测:输出一张标准 markdown 表格,`rates` 单调递增的断言通过,数字和真实项目文档完全一致。

**审稿人会怎么挑刺 + 反驳链:**
- **规模递增轴**:"这个扫描只测到 H=8,更深的想象(H=20、H=50)会不会趋势反转?" → 反�映:如果没测过,
  唯一诚实的回答是承认这是范围声明之外的问题,不能靠已有 5 个点的趋势外推去回答没测过的区间——
  这个边界正是 [05 类](05-limitations-and-honest-disclosure.md)要讲的"局限性诚实自曝"的具体应用
  场景。

**常见坑:**
1. 参数扫描的取值点选得过于稀疏或者不均匀(比如只测 1、2、100 三个点),读者难以判断中间区间的
   真实形状——取值点的选择本身也是实验设计的一部分,需要有覆盖合理区间的考虑。
2. 表格里的数字位数不统一(有的两位小数、有的三位小数)——这种细节不影响科学内容,但会让审稿人
  怀疑整体的严谨程度,统一格式是低成本但容易被忽视的加分项。

---

*上一篇:[02-introduction-and-related-work.md](02-introduction-and-related-work.md)。下一篇:
[04-sentence-level-academic-english.md](04-sentence-level-academic-english.md)——句子层面的学术
英语写作。*
