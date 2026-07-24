# 05 · 图注写作规范深挖(Figure Caption Writing)

> 总览见 [00-roadmap.md](00-roadmap.md)。前四篇讲的都是"怎么把图画出来",这一篇讲图画完之后**配文字**
> 这件事——图注(caption)经常被当成"顺手写一句话"的收尾工作,但 Nature 一类顶级期刊的风格指南明确
> 要求"读者应该能不看正文就理解这张图",这条要求本身可以拆解成几条能被规则检查的具体标准,本篇用真代码
> 把这些标准写成检查函数。

**本篇统一结构(七步):** 签名/是什么 → 一句话 → 底层机制(这里是读者的阅读行为模式)→ AI 研究场景 →
可运行例子 → 审稿人/读者会怎么挑刺 → 常见坑。

**只有 3 个知识点**:图注写作判断力本身的分支不像颜色/排版那样天然对应很多个独立的技术子问题,3 个
知识点分别对应"自洽性要素""标题怎么写""多面板怎么组织",讲透了就不再拆分凑数。

---

## 1. 自洽性原则——caption 要能脱离正文被理解

**是什么:**
```
check_caption(caption: str, has_error_bars: bool) -> list[str]   # 返回发现的问题列表,空列表表示通过
```

**一句话:** 一条合格的图注,读者不看正文就应该能看懂这张图在说什么——具体拆解成几条可检查的标准:
开头是不是在陈述一个结论(而不是泛泛描述"这是一张什么图")、如果图上有误差棒是不是说明了它代表什么、
篇幅是不是长到真的能自成一段说明,这几条标准里至少前两条,可以直接写成规则去检查一段文字。这条原则和
[research-writing-deep-dive/03 号文件"常见坑"第 2 条](../research-writing-deep-dive/03-method-and-results-presentation.md)
提到的"图注只写'如图所示'"是同一件事的两个角度:那边讲"为什么这是个坑",这里讲"具体怎么检查、怎么写对"。

**底层机制/为什么这样设计(这里是读者的阅读行为模式):** 审稿人和很多真实读者的阅读顺序,并不是从
Abstract 一路线性读到 Conclusion——很常见的模式是先把所有图表连同图注快速过一遍,再决定要不要回头
细读正文的具体段落。这意味着图注实际上要承担"如果读者只看图和图注,也能获取这张图的核心结论"这个
职责。Editage 等学术写作指南把一条合格 caption 的要素总结为:一句陈述式标题(说明这张图证明了什么,
不是单纯描述"画的是什么")、必要的方法学上下文(用了什么协议/样本量)、以及统计信息的定义(误差棒
代表什么、多少次重复/多少个随机种子)。反过来,"避免把 Methods 部分大段照抄进图注"同样是这些指南
强调的另一面——图注的目标是"读者理解这张图所需的最小信息量",不是完整的方法学重复。

**AI 研究场景:** 审稿人在有限时间内评审一篇论文,图和图注往往是除了摘要之外获得关注最多的部分——
一条含糊的 caption(比如只写"Success rate under different settings.")几乎没有提供任何超出图本身的
信息,读者还是要自己回正文找上下文,这种"图注没有真正帮上忙"的情况在赶时间的审稿阶段会直接拉低对
论文严谨程度的第一印象。

**可运行例子:**
```python
import re

VAGUE_OPENERS = ["shows", "plot of", "graph of", "figure of", "illustration of"]

def check_caption(caption, has_error_bars):
    """检查一条图注是否满足自洽性的几条基本要求。返回问题列表,空列表表示全部通过。"""
    issues = []
    lowered = caption.strip().lower()
    if any(lowered.startswith(o) for o in VAGUE_OPENERS):
        issues.append("opens with a purely descriptive phrase instead of a declarative finding")
    if has_error_bars and not re.search(
        r"error bar|standard (deviation|error)|\bci\b|confidence interval|\bstd\b", lowered
    ):
        issues.append("figure has error bars but caption never defines what they represent")
    if len(caption.split()) < 8:
        issues.append("too short to be self-contained (fewer than 8 words)")
    return issues

bad_caption = "Plot of the adaptive controller's success rate across every imagination budget level we tested."
good_caption = ("Adaptive imagination budget allocation improves task success rate over both fixed "
                "baselines. Error bars show standard deviation across 5 random seeds.")

bad_issues = check_caption(bad_caption, has_error_bars=True)
good_issues = check_caption(good_caption, has_error_bars=True)
assert bad_issues == [
    "opens with a purely descriptive phrase instead of a declarative finding",
    "figure has error bars but caption never defines what they represent",
]
assert good_issues == []
print("bad caption issues:", bad_issues)
print("good caption issues:", good_issues)
```

**审稿人/读者会怎么挑刺:**
- "这张图的标题只说'不同设置下的成功率',具体哪个设置更好、好多少,我得回正文才能确认。"——图注没有
  提供任何超越图本身的信息,读者感受到的是"这一段没有帮上忙",审稿人对这种"信息真空"的图注格外敏感。
- "误差棒代表什么完全没说明,我怎么判断这个差距是不是噪声?"——[01 号文件第 2 条]
  (01-chart-type-selection.md)已经讲过误差棒本身怎么画,图注里补上定义是最后一环,漏掉这一环前面
  画对了也不算完整。
- "图注这一段几乎是把 Methods 3.2 节复制粘贴过来的。"——过度详尽同样是问题,图注该保留的是"理解这张
  图所需的最小信息",不是把方法学细节重复一遍,占用宝贵的图注篇幅还降低了可读性。

**常见坑:**
- 用"Plot of X vs Y""Comparison of A and B"这类纯描述式开头,没有陈述任何结论——[下面第 2 条]
  (#2-陈述式标题-vs-描述式标题一图一个故事怎么落到文字)会展开讲怎么把这类标题改写成陈述式。
- 图上明明画了误差棒,但图注只字未提是标准差还是标准误、是跨几个随机种子算出来的——这条呼应
  [01 号文件第 2 条](01-chart-type-selection.md)提到的 NeurIPS 检查清单要求,图注和正文都要说明
  误差棒的具体含义。
- 图注写得极短(比如只有三五个词),看起来像是"标题"而不是"图注"——真正合格的图注篇幅通常需要
  一到三句话才能把陈述式标题+必要上下文+统计信息都容纳进去,过短往往意味着信息不完整。

---

## 2. 陈述式标题 vs 描述式标题——"一图一个故事"怎么落到文字

**是什么:**
```
classify_title(title: str) -> "declarative" | "descriptive"
```

**一句话:** 描述式标题("Plot of success rate vs budget")只说了"这张图画的是什么",陈述式标题
("Adaptive imagination outperforms fixed baselines")直接说出"这张图证明了什么"——后者是当前学术
写作指南更推荐的方向,因为它把"一图一个故事"这条原则,从抽象的设计理念变成了标题这一行文字里能不能
读出一个明确论断的具体检验标准。

**底层机制/为什么这样设计:** 学术写作指南记录了一个明确的历史演变:传统上图注标题只标出"图里画的是
哪些变量"(比如"散点图展示了 X 和 Y 的关系"),现代实践越来越倾向于让标题本身就陈述这张图支撑的具体
结论。这个转变背后的逻辑很直接——如果一张图的设计初衷就是服务于论文的某个具体论点([01 号文件]
(01-chart-type-selection.md)"table vs figure"判断里已经强调过"能一句话说清楚就不需要图表"的反面),
那么这个论点应该能被直接写成标题的那一句话,而不是让读者自己从图里的曲线形状反推出来。这不是要求
标题浮夸或者过度总结,是要求标题承担起"陈述这张图的核心发现"这个职责,区别于仅仅"描述图的构成元素"。

**AI 研究场景:** 论文核心结果图([07 号教程体文件](07-build-a-mini-publication-figure.md)会完整
实践这一步)的标题,应该是全文最容易被审稿人快速抓住核心论点的一句话之一——如果一个只花 30 秒扫过
图表部分的审稿人,读完这一句标题就能说出这篇论文的核心贡献是什么,这条标题就是合格的;如果读完标题
只知道"这张图画了几条曲线",没有获得任何论点信息,就还停留在描述式阶段。

**可运行例子:**
```python
DESCRIPTIVE_OPENERS = ["plot of", "graph of", "comparison of", "illustration of", "figure showing", "shows"]
RESULT_VERBS = ["improves", "increases", "decreases", "reduces", "outperforms", "reveals",
                "enables", "hurts", "fails", "only pays off", "beats"]

def classify_title(title):
    """一个足够朴素但能覆盖常见场景的启发式分类器:标题里出现表示'结果/结论'的动词,
    且不是以纯描述式短语开头,判定为陈述式;否则判定为描述式。"""
    lowered = title.strip().lower()
    starts_descriptive = any(lowered.startswith(o) for o in DESCRIPTIVE_OPENERS)
    has_result_verb = any(v in lowered for v in RESULT_VERBS)
    if has_result_verb and not starts_descriptive:
        return "declarative"
    return "descriptive"

t1 = "Plot of success rate versus imagination budget."
t2 = "Adaptive, task-conditioned imagination outperforms both fixed baselines."
t3 = "Imagination only pays off with a task-relevant information edge."

assert classify_title(t1) == "descriptive"
assert classify_title(t2) == "declarative"
assert classify_title(t3) == "declarative"
print(f"{t1!r} -> descriptive")
print(f"{t2!r} -> declarative")
print(f"{t3!r} -> declarative")
```

**审稿人/读者会怎么挑刺:**
- "看完这个标题,我完全不知道你们的方法到底是更好还是更差。"——描述式标题最直接的失败模式:提供了
  "这张图画了什么"的信息,却没有回答"所以呢"这个更重要的问题。
- "标题说'某某方法优于基线',但我在图里没有一眼看出统计显著性支持这个论断。"——陈述式标题的代价是
  它做出了一个更强的断言,图本身(以及误差棒,呼应[本文第 1 条](#1-自洽性原则caption-要能脱离正文被理解))
  必须有足够的证据撑住这句话,不能是标题喊得响、图本身证据不够扎实。

**常见坑:**
- 把陈述式标题写得过度夸张、超出图本身实际支撑的结论强度(比如图里只测了一个任务场景,标题却写成
  普遍性断言)——陈述式标题的信息密度更高,相应地也对"这句话有没有被图完整支撑"提出了更高的要求,
  不是换个句式就万事大吉。
- 多面板图的每个面板各自有自己的描述式小标题,但整张图缺少一个统领全局的陈述式主标题——
  [下面第 3 条](#3-多面板图的-caption-组织统领性主标题--每个-panel-单独-legend)专门讲这个组织问题。
- 陈述式标题里塞入过多细节(具体数值、具体消融配置名称),导致标题本身读起来像一句方法描述而不是
  一个干净的论断——核心结论留在标题,具体数值和统计信息留给标题后面的正文说明,两者要有分工。

---

## 3. 多面板图的 caption 组织——统领性主标题 + 每个 panel 单独 legend

**是什么:**
```
check_multipanel_caption(caption: str, panel_letters: list[str]) -> list[str]
```

**一句话:** 多面板图(A/B/C……)的图注需要两层结构:一句不引用任何具体面板编号的统领性陈述式主标题
(说清楚整张图共同支撑的那一个结论),加上每个面板各自的说明——这是 *Nature* 风格指南里明确写出来的
硬性格式要求,不是排版偏好。

**底层机制/为什么这样设计:** *Nature*(及旗下期刊)风格指南原话:图注必须以一句简短的标题开头,这句
标题描述整张图的整体内容、**不能引用某个具体面板**;随后的图例部分要逐个定义每一个面板。这个两层结构
背后的逻辑是:多面板图的存在理由,通常是"几个独立的证据/角度共同支撑同一个结论"([05 号文件第 2 条]
(#2-陈述式标题-vs-描述式标题一图一个故事怎么落到文字)的陈述式标题原则,在多面板场景下被推广成"一个
统领性结论"),如果主标题直接引用"面板 A 展示了……",相当于把"整体结论"降级成了"某一个面板自己的
描述",读者读完主标题反而搞不清楚这几个面板合起来到底想说明什么。

**AI 研究场景:** [07 号教程体文件](07-build-a-mini-publication-figure.md)最终画的核心结果图是双
面板结构(性能曲线 + 汇总对比柱状图),图注需要一句不提"面板 A""面板 B"的主标题(陈述"想象只有在
获得任务相关信息优势时才划算"这一整体结论),再分别用 "(A) ……" "(B) ……" 说明每个面板具体画了
什么——这个结构本身也是多面板图设计阶段就要想清楚的事,不是画完图之后才临时拼凑图注。

**可运行例子:**
```python
import re

def check_multipanel_caption(caption, panel_letters):
    """检查:①标题句(caption 的第一句)不引用任何面板字母 ②每个面板字母都有形如 "(X) ..." 的
    独立说明。返回问题列表,空列表表示全部通过。"""
    issues = []
    sentences = caption.strip().split(".")
    title_sentence = sentences[0] if sentences else ""
    for letter in panel_letters:
        if re.search(rf"\b{letter}\b", title_sentence):
            issues.append(f"title sentence references panel {letter} -- title should describe the whole figure")
    for letter in panel_letters:
        if not re.search(rf"\({letter}\)", caption):
            issues.append(f"panel {letter} has no its own explanatory clause like '({letter}) ...'")
    return issues

good_caption = ("Imagination only pays off with a task-relevant information edge. "
                "(A) Success rate as a function of imagination budget for three conditions. "
                "(B) Aggregate success rate at the largest budget tested, same colors as (A).")
bad_caption = "Panel A shows the curve and panel B shows the bar chart."

assert check_multipanel_caption(good_caption, ["A", "B"]) == []

bad_issues = check_multipanel_caption(bad_caption, ["A", "B"])
assert len(bad_issues) == 4   # 标题句提到了A和B(违反①),且两个面板都没有 "(X) ..." 格式的独立说明(违反②)
print("bad caption issues:")
for issue in bad_issues:
    print(" -", issue)
print("good caption: passes all checks")
```

**审稿人/读者会怎么挑刺:**
- "主标题里写'面板 A 展示了 XX',那面板 B 的内容跟这个主标题是什么关系?"——主标题引用了单个面板,
  读者会困惑"那另外几个面板是不是次要的/不重要的",破坏了多面板图"共同支撑一个结论"的设计初衷。
- "面板 C 完全没有单独的文字说明,我怎么知道这个子图具体在画什么?"——遗漏某个面板的独立说明,是
  多面板图注最常见的疏漏,面板越多越容易在写图注时漏掉最后一两个。
- "(A)(B)(C) 在图上出现的顺序和图注里说明的顺序不一致。"——图注里各面板说明的顺序应该和它们在
  图上的实际排列顺序(通常是从左到右、从上到下)一致,顺序错位会让读者需要来回对照才能确认。

**常见坑:**
- 主标题不小心引用了某个面板编号(比如"如图 A 所示,……"这种句式直接出现在了标题句里),违反
  *Nature* 风格指南"标题不引用具体面板"的硬性要求。
- 面板数量增多后(比如四五个面板),图注文字量线性增长,容易写成一大段不分段落的密集文字——可以用
  分号或者显式的 "(A) ... (B) ... (C) ..." 结构保持视觉上的可扫描性,不要让读者在一整段无结构的文字
  里自己去找"C 面板到底是哪一句"。
- 面板字母在图上的标注(比如 [03 号文件第 1 条](03-multi-panel-layout-engineering.md)`gridspec`
  例子里 `set_title(letter, loc="left")` 画出来的 A/B/C)和图注里引用的字母大小写/顺序不一致——
  这种细节不一致虽然不影响理解,但是审稿人核对时容易被当成"论文不够仔细"的印象分扣分项。

---

*创建:2026-07-25*
