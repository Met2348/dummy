# 06 · 多轮修改方法论与 AI 辅助边界(Revision Methodology & AI Boundary)

> 总览见 [00-roadmap.md](00-roadmap.md)。这一篇讲"怎么把一篇写完的草稿变成一篇能投的论文",以及一个
> 2023 年之前完全不存在、现在绕不开的新判断题:哪些修改工作可以交给 AI,哪些必须自己把关。

---

## 1. Revision 不是 Editing:先动骨架,再动措辞

写作研究者普遍区分两个阶段:**revision(修订)**处理内容、结构、论证顺序——是不是该拿掉一整段、
两节顺序要不要对调、某个论点的证据够不够;**editing(编辑)**处理句子表面——语法、用词、标点。
常见的低效模式是两个阶段混在一起做,一边逐句抠用词一边试图判断"这一段是不是该挪到别处",这样做
效率低,而且容易在"这句话用词准不准确"这种细节上花掉本该用来判断"这一段到底该不该存在"的精力。
更有效的顺序是先做完至少一轮 revision(骨架级别的增删调整),确认内容和结构基本稳定之后,再进入
逐句 editing。

**常见误区/反面例子:** 从头到尾"一边读一边改",读到哪句顺手改哪句的措辞,改到最后发现某一整段其实
和后面重复、该删,却已经在这段上花了大量时间抠字眼——这些字眼层面的功夫在段落被删除的那一刻全部
浪费。

**逐处修改对照(这里对照的是修改动作本身,不是一段文字):**

| 反例流程 | 问题 | 改法 |
|---|---|---|
| 从第一句开始逐句精修用词,同时判断段落取舍 | 骨架级决策(留/删/挪)和措辞级决策混在一起,骨架变动会
  让已经花的措辞功夫作废 | 先通读全文只做骨架级判断(见知识点 2 的"反向提纲"),骨架稳定后再逐句
  editing |
| 每次改完就直接定稿 | 没有量化过"这次改动到底改了多少",容易低估或高估修改幅度 | 用工具客观量化
  改动幅度(见下方代码),而不是凭印象判断"这次改得够不够" |

**可操作检查清单:**
- [ ] 是否明确分了至少两轮:第一轮只做骨架级判断(段落去留、顺序调整),第二轮才逐句抠用词
- [ ] Revision 阶段是否敢于整段删除已经写好的内容——舍不得删是这个阶段最大的心理障碍,但保留一段
  "写得不错但和主线无关"的内容,对读者是负担不是加分
- [ ] Editing 阶段是否覆盖了 [04 类](04-sentence-level-academic-english.md)讲的具体检查项(被动
  语态、套话、名词化),而不是凭语感随便扫一遍

**量化验证:** "改得好不好"是判断力问题,但"改动幅度有多大"是可以用标准库精确计算的——`difflib`
可以给出改动前后的相似度和具体编辑操作,把"我这次改了很多"从一句主观感受变成一个可以复核的数字:

```python
import difflib

draft = (
    "It is worth noting that the controller's performance is improved by the addition of "
    "task information. The baseline was outperformed in all cases that were tested."
).split()

revised = (
    "Task information improves the controller's performance. Our method outperforms the "
    "baseline in every setting we tested."
).split()

sm = difflib.SequenceMatcher(None, draft, revised)
ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
ratio = sm.ratio()

print(f"draft word count={len(draft)}, revised word count={len(revised)}, similarity ratio={ratio:.2f}")
print(f"non-equal edit ops={len(ops)}")
for tag, i1, i2, j1, j2 in ops:
    print(f"  {tag}: draft[{i1}:{i2}]={draft[i1:i2]} -> revised[{j1}:{j2}]={revised[j1:j2]}")

assert ratio < 0.7  # 改动幅度应该明显,不是微调
assert len(ops) >= 3
print("OK: revision magnitude quantified with SequenceMatcher, not just claimed as 'changed a lot'")
```

本机实测:相似度 `ratio=0.29`(大幅改写,不是微调),5 处非 equal 编辑操作,涵盖删除套话("It is
worth noting that")、语态转换(被动→主动)、措辞精简("all cases that were tested"→"every setting
we tested")。**这个工具的实际用途**不是要求每次修改都必须让 ratio 掉到某个阈值以下——而是当自己
感觉"这次好像没怎么改"但 ratio 显示改动其实很小的时候,提醒自己是不是在回避真正需要的骨架级调整。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"相似度低就等于改得好吗?会不会只是改了很多无关紧要的地方,核心问题反而没碰?"
  → 反驳:完全可能——`ratio` 只衡量"文本变化了多少",不衡量"变化是否命中了真正的问题",这是这个
  工具的真实边界;它能回答"我是不是只做了字面微调",不能回答"我改的是不是该改的地方",后者依然
  需要人工判断。

**常见坑:**
1. 骨架级修改和措辞级修改反复交替、来回横跳,每次都要重新适应新的段落顺序再抠字眼——效率损耗
   累积起来非常可观,严格分阶段能省下大量重复劳动。
2. 误以为"改动越大越好",为了制造"看起来改了很多"的效果做大量无意义的同义词替换——工具能告诉你
   改动幅度,但幅度大小从来不是目标本身,目标是内容和论证质量。

---

## 2. Read Aloud 与 Reverse Outline:两个低成本、高命中率的自检技巧

**朗读(read aloud)**是多个独立写作指南反复推荐的技巧——读出声音(或者用工具朗读)会强迫眼睛放慢
速度,让人更容易发现"写的时候没意识到,但读出来立刻卡壳"的长难句、主谓不一致、重复用词。它对
[04 类](04-sentence-level-academic-english.md)讲的"stress position 被埋没"问题尤其有效:一句话
读到一半喘不上气,几乎总是句子该拆分的信号。

**反向提纲(reverse outline)**是另一个技巧:写完之后,只摘出每一段的第一句话(topic sentence),
拼起来单独读一遍。如果这份"提纲"读起来逻辑顺畅、看得出论证的递进关系,说明段落组织没问题;如果
读起来东一句西一句,说明问题不在措辞,而在段落顺序或者段落本身该不该存在——这正好和知识点 1 讲的
"先做骨架级判断,再做措辞级判断"衔接起来:反向提纲就是骨架级判断的具体操作方法。

**常见误区/反面例子:** 只在脑内默读一遍就认为"读过了",默读时大脑会不自觉地按照"我知道我想表达
什么"去脑补,补上写作时漏掉的连接词和过渡,导致真正的问题被跳过没发现。

**逐处修改对照:** 把"默读一遍"换成"念出声"或者"只读每段第一句拼起来的提纲",两者都是用**改变
阅读方式本身**来强迫自己用读者的视角重新过一遍文本,而不是继续用作者视角自动脑补。

**可操作检查清单:**
- [ ] 定稿前是否至少朗读过一遍全文(或者关键段落),而不是只默读
- [ ] 是否单独抽取过一份"反向提纲"(每段第一句话),通读这份提纲判断段落顺序是否合理
- [ ] 朗读时卡壳的句子是否都回头处理了,而不是当下意识到"这里有点绕"但读完就忘了

**量化验证:** "朗读是否发现了真问题"没法用代码验证,但"反向提纲"本身是一个可以完全自动化的具体
操作,下面的代码演示如何机械地提取每段第一句话,生成的提纲需要读者自己通读判断,工具只负责"提取"
这个体力活:

```python
import re

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def reverse_outline(document):
    paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
    outline = []
    for i, p in enumerate(paragraphs, 1):
        sents = split_sentences(p)
        topic_sentence = sents[0] if sents else ""
        outline.append({"paragraph": i, "topic_sentence": topic_sentence})
    return outline

doc = (
    "World models let an agent imagine future outcomes before acting. This has been used in "
    "many recent systems.\n\n"
    "However, existing controllers spend a fixed budget regardless of need. We show this can "
    "hurt performance when the rollout shares the baseline's model.\n\n"
    "We propose a controller that allocates budget based on estimated information advantage. "
    "It reaches 82.0% hit rate versus 63.7% for the unconditioned baseline."
)

outline = reverse_outline(doc)
for item in outline:
    print(f"Paragraph {item['paragraph']}: {item['topic_sentence']}")

assert len(outline) == 3
assert outline[0]["topic_sentence"].startswith("World models")
print("\nOK: reverse outline = topic sentences concatenated, one read reveals whether the flow holds together")
```

本机实测:三段提纲拼起来是"World models let an agent imagine... / However, existing controllers
spend a fixed budget... / We propose a controller that allocates budget..."——单独读这三句话,
"背景 → 转折出缺口 → 我们的方案"这条主线清晰可辨,说明这三段的顺序和各自的开篇句是称职的。**明确
边界**:这个工具只做机械提取,不判断提纲本身逻辑是否顺畅——这一步永远需要人工通读判断,自动化能
省的是"逐段翻回去找第一句话、手动拼接"这个体力活,不是判断力本身。

**审稿人会怎么挑刺 + 反驳链:**
- **决策依据追问轴**:"如果反向提纲读起来很顺,是不是就说明文章结构没问题了?" → 反驳:不能这么
  确定——反向提纲只检查"段落主线是否连贯",查不出"某一段内部的论证是否站得住"或者"某个具体数字
  是否用错了地方"这类问题,它是一个筛查骨架问题的工具,不是全面质量保证。

**常见坑:**
1. 反向提纲只做一次就再也不重复——修改论文本身经常导致段落顺序变化,每次做较大改动后都值得重新
   抽取一遍反向提纲,不是一次性的检查。
2. 朗读时只关注"读起来顺不顺",忽略了朗读同时也是发现事实性错误的好机会(比如一个数字念出来突然
   意识到"这个数字好像不对")——朗读的价值不局限于语感,也包括慢下来重新审视内容本身。

---

## 3. 找圈外人读:能挑出"我以为讲清楚了但其实没有"的地方

多个写作指南一致建议:找一个**不熟悉这个具体研究话题**的人通读草稿,给出反馈。这条建议背后的逻辑
很直接——作者自己对论证过程太熟悉,容易产生"这一步显然成立"的错觉,而这个"显然"往往是没有被真正
写清楚、只是作者自己脑内脑补出来的连接。一个圈外人(哪怕是同校不同方向的研究生)读到卡壳的地方,
几乎必然对应着论证链条里一个真正的跳跃。

**常见误区/反面例子:** 只请同一个课题组、每天讨论这个项目的同学看草稿——同组的人对项目背景太熟悉,
读的时候会自动脑补作者没写清楚的部分,和作者本人犯的是同一种"过度熟悉导致盲区"的错误,提不出真正
有效的反馈。

**逐处修改对照:** 请一位不了解这个具体子问题、但有基本科研训练的人读——给他们具体的引导性问题
("这一段论证你能不能复述一遍你理解的逻辑""哪一句你需要读两遍才能懂"),而不是笼统地问"你觉得
写得怎么样"(笼统问题容易得到笼统、不具体可操作的反馈)。

**可操作检查清单:**
- [ ] 定稿前是否找过至少一位不熟悉这个具体子问题的人通读过草稿
- [ ] 给对方的引导性问题是否具体(比如"这个论证哪里你没看懂"),而不是泛泛地问"感觉如何"
- [ ] 反馈者本人不应该直接改写文字(这是审稿人自己在做修改,不是给反馈)——他们的角色是**指出**
  哪里不清楚,具体怎么改依然是作者自己的判断力工作,不能外包出去

**量化验证:** 这是一个纯人际协作流程,没有代码可以验证"找没找过圈外人看"这件事,也不该假装能
验证——如实标注这是判断力/工作习惯问题,唯一能做的量化提醒是:如果反馈者读到某一段,复述出来的
理解和作者本意有出入,这本身就是一个客观发生过的事件(可以记录下来:"第 3 段被误读成 X,本意是
Y"),这类记录积累起来能帮助识别自己写作里反复出现的、容易被误读的表达模式。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"找圈外人看,真的能替代找同行审稿人模拟审稿吗?" → 反驳:不能完全替代,两者
  检验的是不同层面的问题——圈外人能有效检验"这段文字本身是否自洽、清楚",但检验不出"这个方法和
  某篇圈外人根本不知道的相关工作是否冲突"这类需要领域知识的问题,后者需要找真正熟悉该子领域、但
  不是项目组成员的人(理想情况下模拟一次内部"预审"),两种反馈来源互补,不是二选一。

**常见坑:**
1. 把"找人看"这件事拖到投稿前最后一两天才做,收到反馈时已经没有时间做出实质调整——早一点找反馈,
   哪怕草稿还不完整,也比赶在截止日期前匆忙塞给别人看更有价值。
2. 收到反馈后选择性采纳(只接受和自己已有判断一致的意见,忽略真正戳中盲区的意见)——找反馈的
   价值恰恰在于暴露作者自己看不到的盲区,如果只挑顺耳的采纳,等于没找。

---

## 4. AI 辅助润色的边界:哪些能用,哪些必须自己把关

2023 年之后,"用不用 AI 辅助写作"从一个可选项变成了几乎每篇论文都要面对的真实判断题,而各大会议
的官方政策在这几年里持续演化,而且彼此不完全一致。基于真实调研到的官方政策(不同年份、不同会议
之间存在差异,投稿前务必核对目标 venue **当年**的最新政策,不能想当然套用别的会议或往年的规则):

| 会议/年份 | 核心态度 |
|---|---|
| ICML 2023 | 禁止提交完全由大模型生成的论文,但允许用 AI **编辑和润色**已有文字 |
| ICML 2026 | 允许用生成式 AI(含 LLM)辅助写作或研究,但要求遵守格式/风格规范,不合规会被直接拒稿 |
| ICLR 2025 / 2026 | 任何 LLM 使用都必须**披露/致谢**,作者对 AI 生成的内容依然负完全责任;明确禁止
  "prompt injection"(在论文里藏对审稿 AI 的提示注入攻击);违反可能导致 desk reject |
| NeurIPS 2025 | 允许把 LLM 当工具使用,但如果 LLM 被用作**核心方法的一部分**,需要在论文里详细描述
  这个使用方式 |
| CVPR 2025 | 明确禁止用 AI 撰写审稿意见(这条是审稿人政策,不是作者写作政策,不要混淆) |

**这些政策共同收敛出一条比较稳定的判断力边界**(不是某一家会议独有,是多家政策的共同倾向):
**用 AI 做语法/表达层面的润色**(检查语法错误、建议更清晰的措辞、翻译)普遍被允许;**用 AI 生成
核心论证内容、实验结论、或者未经验证的技术判断**,普遍要求披露,部分场景明确禁止或要求详细说明。
换句话说:AI 可以帮你把"我想说的话"说得更清楚,不能替你决定"该说什么"。

**常见误区/反面例子:** 把 AI 生成的整段方法论述或实验分析文字直接复制进论文,既没有验证其中的
技术判断是否准确,也没有按目标 venue 要求披露使用情况——这不只是学术诚信风险,也是真实的技术
风险:AI 生成的技术性陈述可能包含听起来合理但实际不准确的内容,一旦被审稿人识破,后果比语法错误
严重得多。

**可操作检查清单:**
- [ ] 明确投稿前核实了目标 venue **当年**的 LLM 使用政策原文,不是凭印象套用别的会议或者往年的规则
- [ ] AI 参与的部分,是不是能清楚归入"表达润色"而不是"内容生成"——判断标准是"这段话的技术判断
  是不是我自己先想清楚、AI 只是帮我表达得更好",还是"这段话的技术判断本身是 AI 替我做的"
- [ ] 如果目标 venue 要求披露,是否已经按要求在论文里加入了对应的致谢/声明
- [ ] 任何 AI 生成或润色过的技术性陈述(尤其是涉及数字、引用、方法机制的部分),是否逐一人工核实过
  ——这条呼应知识点 5 讲的"二次核验",AI 辅助不能替代这一步

**量化验证:** "这段文字算润色还是算内容生成"本身是判断力问题,不可 assert;但"给定的使用场景,
在特定 venue 的简化规则下是否合规"是可以用规则引擎检查的——下面的代码把三家会议的政策编码成
一个简化的规则表(注意这是教学用的**简化版本**,不是完整法律文本,真实投稿必须以官方原文为准):

```python
# 简化编码几家会议2023-2026年公开政策的核心区别(不代表完整法律文本,仅作自查用的简化规则)
VENUE_RULES = {
    "ICML_2023": {"allow_grammar_polish": True, "allow_content_generation": False, "require_disclosure": False},
    "ICML_2026": {"allow_grammar_polish": True, "allow_content_generation": True, "require_disclosure": True},
    "ICLR_2025_2026": {"allow_grammar_polish": True, "allow_content_generation": True, "require_disclosure": True},
    "NeurIPS_2025": {"allow_grammar_polish": True, "allow_content_generation": True, "require_disclosure": True},
}

def check_compliance(venue, used_for_grammar, used_for_content, disclosed):
    rule = VENUE_RULES[venue]
    violations = []
    if used_for_content and not rule["allow_content_generation"]:
        violations.append("venue does not allow LLM-generated core content (polish only)")
    if (used_for_grammar or used_for_content) and rule["require_disclosure"] and not disclosed:
        violations.append("venue requires disclosing LLM use but it was not disclosed")
    return {"compliant": len(violations) == 0, "violations": violations}

case_2023_undisclosed_content = check_compliance("ICML_2023", used_for_grammar=True,
                                                   used_for_content=True, disclosed=False)
case_2026_disclosed = check_compliance("ICML_2026", used_for_grammar=True,
                                        used_for_content=True, disclosed=True)
case_2026_undisclosed = check_compliance("ICLR_2025_2026", used_for_grammar=True,
                                          used_for_content=False, disclosed=False)

print("ICML_2023, used for content generation and undisclosed:", case_2023_undisclosed_content)
print("ICML_2026, used for content generation but disclosed:", case_2026_disclosed)
print("ICLR_2025_2026, used only for grammar polish but undisclosed:", case_2026_undisclosed)

assert case_2023_undisclosed_content["compliant"] is False
assert case_2026_disclosed["compliant"] is True
assert case_2026_undisclosed["compliant"] is False
print("OK")
```

本机实测:三种场景的合规判断和政策原文对照全部一致。**明确边界**:这份 `VENUE_RULES` 是本篇写作
时基于 WebSearch 调研到的政策摘要做的简化编码,几个字段(尤其是"content_generation"这种边界本身
就模糊的概念)在真实政策文本里往往有更细致的限定条件,这个工具只适合当作"投稿前记得去查官方原文"
的提醒清单,不能当作真正的合规判定依据——真正投稿前必须逐字重读目标 venue 当年的官方政策页面。

**审稿人会怎么挑刺 + 反驳链:**
- **规模递增轴**(适配成"政策的时效性"):"你这张表里的政策是 2025-2026 年的,读者看到这篇笔记的
  时候政策是不是早就变了?" → 反驳:几乎肯定会变——这是这类"引用会议官方政策"内容天然的过期
  风险,00 类 roadmap 和这里都需要提醒读者:任何具体政策条款都应该以投稿当时的官方页面为准,这篇
  笔记的价值是"教会你这类政策通常在区分什么"(润色 vs 内容生成、是否要求披露),不是提供一份可以
  长期有效的规则速查表。

**常见坑:**
1. 假设所有会议的政策相同,用一家的规则套用到投稿的另一家会议——这是这里明确调研到的真实差异
   (比如 ICML 2023 和 ICML 2026 之间政策就有明显演化),混用会导致真实的合规风险。
2. 把"AI 帮我把语法改对了"和"AI 帮我把论证补完整了"混为一谈——前者几乎在所有政策下都被允许,
   后者是真正需要谨慎对待、大概率需要披露甚至可能被禁止的场景,两者的边界需要作者自己诚实判断,
   不能因为"反正 AI 生成的东西读起来很像我自己写的"就模糊处理。

---

## 5. 真实案例:二次核验订正——多轮修改真正抓住的是事实错误,不只是语法

多轮修改经常被误解成"改善措辞",但真正高价值的修改往往是**核实事实性内容**——引用的数字对不对、
论文机制复述准不准确、图表符号有没有误读。`research/world-model-imagination-controller/
01-meeting-briefing.md` 里明确标注了好几处"二次核验订正",是这次撰写过程里能找到的最好的真实范例
(按设计文档要求,这里描述的是修改方法论本身,不逐字复制原文的技术结论):

1. **数字精度订正**:项目组引用 FFDC 论文里真实机器人的自由度数字时,第一版写成了"25 自由度",
   二次核验回去核对原文后发现真实数字是"34 自由度",已订正。这是最基础也最容易被忽视的一类错误——
   任何引用的具体数字,在定稿前都值得回去和原文再核对一遍,记忆里的数字经常会在传递过程中悄悄
   漂移。
2. **比较力度订正**:项目组最初把自己方法和两个基线(GPT-4o 版本、GPT-4.1 版本)的对比,用同样
   的语气并列陈述,二次核验回去看原文才注意到,原论文对这两组对比的着墨力度并不对等——一组是
   论文正文重点强调的核心对比,另一组只是顺带提及、力度弱得多。这不是数字错误,是**语气/权重
   的误传达**——用同样力度陈述两个原文里权重不同的说法,会不小心夸大其中一个的分量。
3. **不采纳矛盾数据**:项目组在核验 Video-T1 论文一处图表数字时,发现引用的"-2.37"这个数字和
   同一篇论文的另外三处证据(一张表、一张图、正文结论)相互矛盾,大概率是原论文自己图注的符号
   笔误。**最终的处理方式不是"选一个自己更喜欢的版本采用",而是干脆不用这个数字支撑论点**——
   宁可少一个例子,也不复述一个连原始论文自己都自相矛盾的数字。

**可操作检查清单:**
- [ ] 定稿前,是否把论文里引用的每一个具体数字都回去和原始来源核对过一遍,而不是相信自己记忆或
  第一次读到时的笔记
- [ ] 对比多个基线/相关工作时,是否核对过原文对这几个对比的着墨力度是否对等——避免用同样的语气
  陈述原文里权重不同的说法
- [ ] 如果发现某个引用数字和其他可靠来源矛盾,是否选择了"不采用这个数字"而不是"挑一个更符合自己
  论点的版本"——这是学术诚信的底线,不是可以灵活处理的判断力问题

**量化验证:** "这个数字有没有被正确核实"本质是事实核查工作,没有代码能够自动完成——这需要人工
回去比对原始来源,如实标注这是流程纪律问题,不是可以自动化验证的事实。唯一能提供的机械辅助是
知识点 1 的 `difflib` 工具:如果保留了历史版本,可以用它快速定位"这次修改具体改了哪些数字/专有
名词",帮助聚焦核查范围,但"改得对不对"依然要靠回去核对原始来源。

**审稿人会怎么挑刺 + 反驳链:**
- **真实性验证轴**:"你怎么知道自己没有漏掉其他类似的错误,只是恰好抓到了这三处?" → 反驳:诚实
  的答案是不能保证穷尽——二次核验能提高命中率,但不是零错误的保证,这也是为什么"多轮"比"一轮"
  重要:每一轮核验大概率能抓到上一轮漏掉的一部分错误,但没有一个轮次能保证抓完所有错误,这是
  事实核查工作的真实局限,不是这次案例特有的问题。

**常见坑:**
1. 把"二次核验"只用在自己拿不准的地方,跳过"看起来很确定"的内容——很多错误恰恰发生在"我确定
   记得这个数字"的地方,因为确定感本身会降低警惕性,系统性核验应该覆盖全部引用数字,不是凭直觉
   挑着核验。
2. 发现矛盾数据后,选择"两个数字都不提,回避这个对比"而不是诚实处理——上面案例里项目组的处理是
   "不采用这个矛盾数字,但没有回避整体论点",两者是有区别的:前者是诚实应对不确定性,后者是回避
   问题本身,这个界限值得注意。

---

*上一篇:[05-limitations-and-honest-disclosure.md](05-limitations-and-honest-disclosure.md)。
下一篇:[07-reviewer-perspective-and-rejection-patterns.md](07-reviewer-perspective-and-rejection-patterns.md)——
审稿人视角精读,常见拒稿理由拆解。*
