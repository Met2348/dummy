# L5 · 系统性文献综述与 Meta-Analysis 方法论 (Systematic Review & Meta-Analysis Methodology)

> 40-min lecture · 目标: 理解「系统性综述 (systematic review)」和你在 L1-L4 学的「摸地图」是两件性质不同的事——前者的产出本身就是一篇可发表的论文, 后者的产出是给你自己用的导航图。讲清楚支撑这件事的三块方法论: PRISMA 报告规范 (检索策略预注册/筛选标准/双人独立筛选)、PICO(C) 问题拆解框架、以及 meta-analysis 合并效应量时 fixed-effects 和 random-effects 两种模型的区别。
> 关键区分: **L1-L4 教你做的是 scoping review (范围综述)——目标是"两周内自己摸清一个方向的地形", 产出的 mini-survey 是内部工具, 允许你主观取舍、不追求穷尽。本讲教的 systematic review (系统性综述) 是一种正式的研究方法论——它的检索、筛选、合并过程本身必须像一个实验一样可复现、可审计, 产出是提交给期刊/会议的独立成果, 别人可以在不认识你、不信任你个人判断力的前提下, 单凭你写的方法学章节就重新执行一遍你的综述并得到相近的结论。** L1 的三类综述表格里已经把 systematic 列了一行但没展开, 本讲就是展开那一行。

---

## 0. 一个具体场景: 什么时候你需要的不是 mini-survey, 而是一篇 systematic review

设想两个场景, 对比着看:

**场景 A**: 导师说「你去把 in-context learning 这块摸清楚, 两周后来跟我讲」。这正是 9.2 L1-L4 教你做的事——你会找锚点、滚雪球、建地图、写 mini-survey。这份 mini-survey 里, 你可以凭个人判断跳过一些你觉得不重要的论文, 可以只让自己一个人筛选, 没人会去审计你「为什么选了这 40 篇而不是另外 40 篇」——因为它的读者只有你和导师, 目的是给你的下一步研究装 GPS, 不是给学界一个权威结论。

**场景 B**: 你的合作者是医学/公共卫生/教育学背景的博士生, 她所在的领域有一个惯例——想知道「某种干预措施到底有没有用」这类问题, 不会满足于一篇「我读了 20 篇论文, 感觉大概是有用的」narrative review, 而是要求你交出一篇 systematic review + meta-analysis: 检索策略必须写清楚到「在 PubMed 用了哪几个关键词布尔组合、检索日期是哪天」, 纳入排除标准必须写清楚到「谁都能拿着这份标准去把同一批论文重新筛一遍, 筛出来的结果和你差不多」, 如果多篇论文都报告了同一个效应 (比如「某种教学干预让考试分数提高了多少」), 你还要用统计方法把这些数字**合并**成一个综合估计值, 而不是简单地数「几篇说有效、几篇说无效」。

> **本讲要建立的核心区分就是**: L1-L4 的产出 (mini-survey/领域地图) 是**给你自己用的导航图**, 允许主观、允许不完整、允许只有一个人做; systematic review 的产出**本身就是投稿的对象**, 它的每一步都要经得起一个素不相识的审稿人/复现者按你写下的方法学重新走一遍。这和 9.17-L3 讲开放科学时讲的"可核验性"是同一种精神在文献综述这个具体场景下的落地——只是 9.17-L3 讲的是实验数据/代码的可核验, 本讲讲的是"你读了哪些文献、为什么读这些、怎么把它们的结论合并成一个数字"的可核验。

```
   scoping review (L1-L4 已学)              systematic review (本讲)
   ────────────────────────────            ──────────────────────────────
   产出: mini-survey (内部地图)              产出: 可发表的论文本身
   检索: 凭经验滚雪球, 不需要写死           检索: 预注册的检索式, 写死可重复
   筛选: 一个人凭判断取舍                   筛选: 双人独立筛选 + 记录分歧
   问题: 模糊的"摸清 X 方向"                问题: 用 PICO(C) 拆成可检索的字段
   数字: 不合并, 定性描述流派/演进          数字: meta-analysis 统计合并效应量
   时间盒: 2 周                            时间盒: 数月到一年 (要走完整流程)
```

---

## 1. PICO(C): 把一个模糊的研究问题拆成可以拿去检索的四(五)个字段

在动手检索之前, systematic review 要求你先把研究问题拆解成一个标准化的结构, 医学循证领域最常用的框架叫 **PICO**, 全称:

- **P (Population/Problem)** —— 你研究的对象是谁/什么问题? (例如: 患有 II 型糖尿病的成年人; 或者, 正在做数学应用题的中学生)
- **I (Intervention)** —— 你关心的干预/处理是什么? (例如: 每天服用某种药物; 或者, 采用某种新的教学方法)
- **C (Comparison)** —— 拿什么做对照? (例如: 安慰剂/传统教学法/不做任何干预)
- **O (Outcome)** —— 你要衡量的结果指标是什么? (例如: 三个月后的血糖水平; 或者, 期末考试分数)

计算机科学 / 软件工程领域的系统性综述常常在 PICO 后面再加一个 **C (Context)**, 变成 **PICOC**——因为软件工程里"干预"往往严重依赖上下文 (团队规模、项目类型、开发方法论), 这一惯例来自 Kitchenham 与 Charters 2007 年发布的《Guidelines for performing Systematic Literature Reviews in Software Engineering》, 这份指南是把医学系统综述方法论"移植"进软件工程/CS 领域的奠基性文献, 和你已经在 L2 用过的 Wohlin (2014) 滚雪球方法论出自同一个方法论谱系 (软件工程实证研究方法论)。

为什么一定要先拆 PICO(C) 才能检索? 因为一个模糊的问题 (比如"提示工程对大语言模型推理能力有没有用") 没法直接变成检索式, 但拆成 PICO 之后:

```
P: 大语言模型 (在数学推理/常识推理任务上)
I: 某类提示工程技术 (如 chain-of-thought / few-shot prompting)
C: 直接问答 (无特殊提示技术) 作为对照
O: 任务准确率 / 推理正确率

→ 检索式雏形: ("large language model" OR "LLM") AND ("chain-of-thought" OR
             "prompt engineering") AND ("reasoning" OR "accuracy")
```

PICO(C) 每个字段还决定了你后面的**纳入/排除标准** (下一节): 凡是 P 不匹配 (比如论文研究的是图像模型不是语言模型) 或 O 没有报告可比数字的论文, 都会被排除。**没有先拆 PICO(C), 你的纳入排除标准就没有客观依据, 筛选就会变成"我觉得这篇像"的主观判断**——这正是 systematic review 要极力避免的。

---

## 2. PRISMA: 让你的检索和筛选过程本身可以被审计、被重复

**PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses)** 是目前医学/社科/教育/软件工程等多个领域最广泛采用的系统性综述报告规范, 现行版本是 **PRISMA 2020 声明** (Page, McKenzie, Bossuyt 等, 发表于 *BMJ*, 2021)。它不是一种检索工具, 而是一份"你的综述论文的方法学部分必须写清楚哪些条目"的清单 (checklist) + 一张标准化的流程图 (flow diagram)。

### 2.1 检索策略预注册

在正式开始检索之前, 你要把 PICO(C) 拆解出的检索式、准备检索的数据库 (如 PubMed / ACM Digital Library / IEEE Xplore / Semantic Scholar)、检索的起止日期, 全部写下来并**预先登记**——常用的登记平台是 **PROSPERO** (由英国约克大学 Centre for Reviews and Dissemination 运营的国际前瞻性系统综述注册库), 这和你在 9.17-L3「开放科学实践」学过的 OSF 预注册假设是同一种逻辑, 只是登记的对象从"实验假设"换成了"检索策略"——**目的都是把"我打算怎么找证据"这句话的时间戳钉在你看到检索结果之前**, 防止你事后为了凑出想要的结论而悄悄调整检索式或筛选标准 (这是系统综述里的"HARKing 变体": 事后调整方法学去迎合已经看到的结果)。

### 2.2 PRISMA 流程图: identification → screening → eligibility → included

PRISMA 要求你用一张标准化的四阶段流程图报告"多少篇论文从哪个阶段被筛掉、为什么被筛掉":

```
   识别 (Identification)
   数据库检索命中 N1 篇 + 其他来源 (滚雪球/引用检索) N2 篇
              │  去重 (移除 N3 篇重复记录)
              ▼
   筛选 (Screening)
   标题/摘要筛选 N4 篇  ──排除 N5 篇 (标题摘要明显不符 PICO)──►  [排除记录]
              │
              ▼
   资格审查 (Eligibility)
   全文评估 N6 篇  ──排除 N7 篇 (逐条写明排除理由, 如"无对照组"/"结果指标不匹配")──► [排除记录+理由]
              │
              ▼
   纳入 (Included)
   最终纳入 N8 篇进入定性综合和/或 meta-analysis 定量合并
```

这张图的关键价值在于: **每一次"排除"都必须留痕并写明理由**, 读者可以核对"你排除的这些论文, 我要是自己筛, 会不会同意"——这正是 systematic review 和 L1-L4 mini-survey 最大的操作性区别: mini-survey 里你跳过一篇论文不需要理由, systematic review 里跳过一篇必须留下审计记录。

### 2.3 纳入/排除标准与双人独立筛选

纳入/排除标准直接从 PICO(C) 推出 (如: P 不匹配的语言/物种/任务类型排除, I 不是目标干预排除, O 未报告可提取数字排除), 且必须在检索开始前就写死, 不能看完结果再调整。

更关键的方法学要求是**双人独立筛选 (dual independent screening)**: 至少两名评审者各自独立地、互不参考对方判断地对同一批论文做标题/摘要筛选和全文筛选, 筛完之后再对比分歧, 用 **Cohen's kappa 系数**报告两人的一致程度 (kappa 越接近 1 说明标准写得越清晰、筛选越可靠), 分歧的条目提交第三名评审者仲裁。这么做的原因很直接: **单人筛选无法排除"我倾向于纳入支持我预期结论的论文"这种确认偏误 (confirmation bias)**, 双人独立 + 事后核对分歧, 是目前公认最有效的缓解办法。实践中常用的筛选管理工具是 **Rayyan** (Ouzzani, Hammady, Fedorowicz, Elmagarmid, 发表于 *Systematic Reviews* 期刊, 2016)——一个支持"盲筛"模式 (两名评审者互相看不到对方标注) 的网页/移动端工具, 筛选结束后自动统计两人的一致率和分歧列表。

---

## 3. Meta-analysis: 当多篇论文都报告了"同一件事"的数字, 怎么合并成一个可信的估计

PRISMA 流程走完, 你手里有一批通过筛选的论文 (定性综合, qualitative synthesis)。如果其中若干篇报告了**同一种可比较的效应** (比如都报告了"某种干预相对对照组带来的分数提升"), 你可以再往前一步, 做 **meta-analysis (荟萃分析)**——用统计方法把这些独立研究的数字**合并**成一个总体效应量估计, 而不是简单地数"几篇说显著、几篇说不显著" (后面这种做法叫 vote-counting, 是 meta-analysis 明确要避免的粗糙做法, 因为它完全忽略了每篇研究的样本量和精度差异)。

### 3.1 效应量 (effect size): 先把每篇论文的结果换算成同一把尺子

不同论文可能用不同的量纲报告结果 (一篇报告"分数提高了 8 分", 另一篇报告"提高了 15%"), 无法直接相加平均。meta-analysis 第一步是把每篇论文的结果换算成统一的**效应量**指标, 常见的有:

- **标准化均值差 (standardized mean difference, 如 Cohen's d)** —— 用于两组连续型结果的比较 (如干预组 vs 对照组的考试分数), 把"分数差"除以标准差, 消除量纲。
- **比值比 (odds ratio, OR)** / **风险比 (risk ratio, RR)** —— 用于二分类结果 (如"治愈/未治愈"、"通过/未通过")。
- **相关系数 (correlation coefficient, r)** —— 用于两个连续变量之间关系强度的研究。

每篇论文除了效应量本身, 还要带上它的**方差/标准误**——这决定了这篇论文在合并时应该被赋予多大的"话语权重" (样本量越大、估计越精确的研究, 权重越高)。

### 3.2 森林图 (forest plot): meta-analysis 结果的标准可视化

```
   研究            效应量 (95% CI)
   ──────────────────────────────────────
   Study 1    |───●───|              d=0.30 [0.10, 0.50]  权重 15%
   Study 2      |──●──|              d=0.45 [0.25, 0.65]  权重 22%
   Study 3    |────●────|            d=0.20 [-0.05, 0.45] 权重 10%
   Study 4       |─●─|               d=0.38 [0.28, 0.48]  权重 35%
   Study 5     |───●───|             d=0.15 [-0.10, 0.40] 权重 18%
   ──────────────────────────────────────
   合并估计        ◆                  d=0.33 [0.24, 0.42]
                   0        0.5       1.0
```

每一行是一篇研究的效应量点估计 + 置信区间, 菱形 (◆) 是合并之后的总体效应量——这张图让读者一眼看出"各研究结果是否大致一致"、"合并之后的效应有多大、置信区间是否跨过 0 (无效线)"。

### 3.3 Fixed-effects model vs Random-effects model: 两个模型假设的是不同的世界

这是 meta-analysis 里最容易被混淆、也是本讲要讲透的核心概念。两个模型都是"怎么把多篇研究的效应量加权平均成一个数字", 但它们对"这些研究之间为什么结果不完全一样"做了**完全不同的假设**:

- **Fixed-effects model (固定效应模型)**: 假设**只存在一个真实的效应量**, 所有纳入的研究都在估计同一个数, 它们报告的数字之所以有差异, **唯一原因是抽样误差** (每篇研究招募的样本不同, 抽样运气不同)。在这个假设下, 合并权重只由每篇研究的样本量/方差决定 (通常用 inverse-variance weighting, 方差越小权重越大)。
- **Random-effects model (随机效应模型)**: 假设**每篇研究估计的其实是各自不同的真实效应量**, 这些效应量本身来自一个分布 (比如不同研究的干预实施方式、参与人群、测量工具都略有不同, "真实效应"本来就该有差异), 你合并出来的是这个效应量分布的**均值**。除了抽样误差之外, 还要额外估计"研究间方差" (between-study variance, 常记作 τ²), 最常用的估计方法是 **DerSimonian-Laird 方法** (1986 年提出, 至今仍是最广泛使用的默认估计量之一)。随机效应模型的合并权重会比固定效应更"平均" (不会让某一篇超大样本研究完全主导结果), 置信区间通常也更宽 (更保守)。

**怎么判断该用哪个?** 关键看**异质性 (heterogeneity)**——纳入的研究是不是真的在测量"同一件事"。异质性最常用的量化指标是 **I² 统计量** (表示研究间变异中有多大比例不能用抽样误差解释, 0%-100%, 常见的粗略经验分档: <25% 低、25%-75% 中等、>75% 高)。**如果 I² 很低 (研究彼此高度一致), fixed-effects 是合理近似; 如果 I² 较高 (研究之间本身就有实质差异, 比如干预细节、人群、测量方式都不同), 应该用 random-effects, 因为继续假设"只有一个真实效应量"在事实上不成立, 会低估合并估计的不确定性。** 现实中的实证研究 (尤其是跨多个实验室/多种实现细节的研究) 异质性普遍偏高, 因此 **Cochrane Handbook for Systematic Reviews of Interventions** (Higgins 等主编, Cochrane 协作网的方法学圣经) 建议在缺乏充分理由假设"单一真实效应"时, 默认优先考虑 random-effects model。

实践中最常用来跑这些统计的开源工具是 R 语言的 **metafor 包** (Viechtbauer, 发表于 *Journal of Statistical Software*, 2010)——它同时支持 fixed 和多种 random-effects 估计量 (DerSimonian-Laird / REML 等), 内置森林图绘制函数, 是目前学术界做 meta-analysis 事实上的标准工具之一; 医学领域另一常用工具是 Cochrane 自己维护的 **RevMan (Review Manager)** 软件。

---

## 4. 一个完整走查示例

假设你 (或你的合作者) 要做一篇 systematic review, 主题是"某种教学干预对学生数学成绩的影响"——这是循证教育学里的经典 systematic review 场景, 完整走一遍:

```
PICO 拆解:
  P: 中学阶段学生
  I: 某种新教学干预 (如同伴互助学习法)
  C: 传统讲授式教学
  O: 标准化数学测试分数

PRISMA 预注册 (登记在 PROSPERO):
  检索数据库: ERIC / PsycINFO / Web of Science
  检索式: ("peer-assisted learning" OR "peer tutoring") AND
          ("mathematics achievement" OR "math score") AND ("randomized")
  检索日期范围: 2000-01-01 至检索当天, 仅限同行评审期刊/会议论文

PRISMA 流程 (示意数字):
  识别: 数据库命中 842 篇 + 滚雪球 15 篇 = 857 篇, 去重后 610 篇
  标题摘要筛选: 610 篇 → 排除 540 篇 (明显不符 P/I) → 70 篇进入全文评估
  全文评估: 70 篇 → 排除 46 篇 (23 篇无对照组, 15 篇结果指标不可比,
            8 篇非随机分组) → 24 篇最终纳入

双人独立筛选: 两名评审者独立完成标题摘要筛选, Cohen's kappa = 0.81
              (一致性"良好"), 12 处分歧提交第三名评审者仲裁

Meta-analysis:
  24 篇均报告了标准化均值差 (Cohen's d)
  I² = 68% (中高异质性, 干预实施细节/年级/持续时长差异较大)
  → 选择 random-effects model (DerSimonian-Laird 估计量)
  → 合并效应量: d = 0.34, 95% CI [0.21, 0.47], 森林图确认无单一研究主导结果
  结论: 该教学干预对数学成绩有中等程度的正向效应, 但研究间存在实质异质性,
        后续需要亚组分析 (moderator analysis) 探究哪些实施条件效应更强
```

注意这个流程从头到尾**没有一步依赖"我个人觉得"**——检索式提前登记、排除理由逐条留痕、筛选由两人独立完成并报告一致性、合并方法的选择 (fixed vs random) 由客观的 I² 指标决定, 而不是"我想要哪个数字好看就选哪个模型"。这就是 systematic review 相对 mini-survey 多出来的全部工作量, 也是它能作为独立可发表成果的原因。

---

## 5. 和你已有工作流的关系: 什么时候才需要走 systematic review 这条重路

对一名 NLP/LLM 方向的博 0 来说, **绝大多数"摸清一个方向"的日常需求, L1-L4 的 scoping review + mini-survey 完全够用, 不需要走本讲这套重量级流程**。你什么时候才真的需要 systematic review + meta-analysis:

- 你 (或合作者) 要写一篇**以综述本身为贡献**投稿的论文, 而不是把综述当自己研究的前期准备。
- 领域里已经有**足够多结构相近、报告了可比数字**的实证研究 (比如都在报告"某类提示技术带来多少个百分点的准确率提升"), 值得也可能做量化合并——如果每篇论文的实验设定天差地别 (数据集/模型/指标完全不可比), 勉强做 meta-analysis 会得出没有意义的合并数字, 这时定性的 taxonomy + 演进线 (L3 教的方法) 仍然是更诚实的呈现方式。
- 你需要给一个"到底有没有用"式的问题一个经得起审计的答案 (常见于要影响政策/教学实践/临床决策的场合), 而不只是给自己下一步研究定位。

写完的 systematic review 论文本身, 后续的投稿、评审、rebuttal 流程和任何其他论文一样, 接你已经规划要学的 9.7 paper-writing-submission 那一套。

---

## 6. 常见误区

**误区①: 把 narrative review 包装成 systematic review。** 有人在论文里自称"a systematic review", 却没有预注册检索式、没有双人筛选、没有 PRISMA 流程图——这是方法学上的虚假宣称, 审稿人一旦要求提供 PRISMA checklist 和流程图就会立刻露馅。

**误区②: 检索完再回头调整纳入排除标准去凑想要的论文数量/结论。** 这是系统综述场景下的 HARKing 变体——纳入排除标准必须在看到检索结果、更不用说看到论文结论之前就写死, 中途因为发现某类论文"结果不好看"而临时补一条排除标准, 破坏的正是 systematic review 要保证的可核验性。

**误区③: 不管异质性多高, 都无脑用 fixed-effects model 合并 (因为它算出来的置信区间更窄、看起来更"显著")。** 这是选择性使用统计模型来让结果显得更漂亮的做法, 一旦 I² 提示研究间存在实质异质性, 继续假设"只有一个真实效应量"在方法学上是站不住脚的, 应该切到 random-effects 并如实报告更宽的置信区间。

**误区④: 把 vote-counting (数有几篇说显著/不显著) 当成 meta-analysis。** 这种做法完全丢弃了每篇研究的样本量和精度信息, 一篇 5000 人的研究和一篇 20 人的研究被同等计数, 是被 meta-analysis 方法论明确抛弃的过时做法。

---

## 7. 经典参考

- **Page, M. J., McKenzie, J. E., Bossuyt, P. M., et al., "The PRISMA 2020 statement: an updated guideline for reporting systematic reviews", *BMJ*, 2021** —— 目前系统性综述报告规范的现行版本, 本讲第 2 节的核心方法论出处, 定义了检索预注册、四阶段流程图等条目清单。
- **Higgins, J. P. T., Thomas, J., Chandler, J. 等 (主编), *Cochrane Handbook for Systematic Reviews of Interventions*, Cochrane 协作网** —— 系统性综述与 meta-analysis 方法论的权威参考手册, 本讲第 3 节 fixed/random-effects 模型选择的核心依据。
- **Kitchenham, B., Charters, S., "Guidelines for performing Systematic Literature Reviews in Software Engineering", 2007** —— 把系统综述方法论移植进软件工程/CS 领域的奠基指南, PICOC 框架 (在 PICO 基础上加 Context) 的来源, 和 L2 已用过的 Wohlin (2014) 滚雪球方法论同属软件工程实证研究方法论谱系。
- **Viechtbauer, W., "Conducting Meta-Analyses in R with the metafor Package", *Journal of Statistical Software*, 2010** —— meta-analysis 事实上的标准 R 工具, 本讲第 3.3 节 fixed/random-effects 计算的实践参考。
- **Ouzzani, M., Hammady, H., Fedorowicz, Z., Elmagarmid, A., "Rayyan—a web and mobile app for systematic reviews", *Systematic Reviews*, 2016** —— 支持双人盲筛的开源筛选管理工具, 本讲第 2.3 节双人独立筛选的实践参考。
- **PROSPERO** (英国约克大学 Centre for Reviews and Dissemination 运营的国际前瞻性系统综述注册库) —— 检索策略预注册的实践平台, 和 9.17-L3 已讲过的 OSF 假设预注册是同一逻辑在文献综述场景下的对应物。
- 也见本专题 L1「三类综述」表格——本讲展开的是其中被暂时略过的 systematic 一行; L1-L4 教的 scoping review + mini-survey workflow 仍然是你日常摸方向的默认工具, 本讲只在场景 B (综述本身要发表/要给可核验结论) 时才需要。
- 也见本仓库 `open-science-and-communication`(9.17) 专题 L3「开放科学实践」——预注册的通用逻辑 (把承诺的时间戳钉在看到结果之前) 已在那一讲详细展开, 本讲第 2.1 节只是把它对应到"检索策略"这个具体对象上, 不重复展开预注册机制本身。

---

## 8. 本讲小结

- systematic review 和 L1-L4 的 scoping review/mini-survey 是**不同性质**的产出: 前者本身是可发表成果、必须可被陌生审稿人核验重复; 后者是内部导航图, 允许主观取舍。
- **PICO(C)** 把模糊的研究问题拆成 Population/Intervention/Comparison/Outcome(/Context) 四(五)个可检索的字段, 是检索式和纳入排除标准的共同起点。
- **PRISMA** 提供检索策略预注册 (如登记在 PROSPERO)、标准化四阶段流程图 (identification → screening → eligibility → included)、以及双人独立筛选 (报告 Cohen's kappa, 用 Rayyan 等工具执行盲筛) 三件具体的可核验性机制。
- **Meta-analysis** 把多篇研究的效应量 (Cohen's d / OR / RR / r) 加权合并成一个总体估计, 用森林图可视化; **fixed-effects model** 假设只有一个真实效应量、差异全来自抽样误差, **random-effects model** 假设真实效应量本身在研究间有分布、需要额外估计研究间方差 (如 DerSimonian-Laird); 该用哪个由 **I² 异质性指标**客观决定, 而不是挑一个结果好看的。

**动手**: 挑一个你 (或身边合作者) 感兴趣的、已经积累了多篇可比较实证研究的具体子问题 (例如"某类 prompting 技术对某类推理任务的提升幅度"), 用 PICO 框架把它拆成四个字段, 写出一条检索式雏形; 然后设想如果真的收集到 20 篇这样的论文, 你会用什么效应量指标去合并它们, 以及你预期 I² 会偏高还是偏低——把这个思维实验写成不超过 200 字的纪要, 体会"拆 PICO"和"选合并模型"这两步具体在做什么决策。
