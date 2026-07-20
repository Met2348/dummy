# L10 · 非英语母语者的学术写作策略: 中式英语陷阱与写作辅助工具的正确用法

> 30-min lecture · 目标: L1-L9 全程假设你已经能自如地把论证逻辑写成流畅英语。但如果英语不是你的母语, **写作本身就多了一层翻译损耗**——本讲不是语法课, 而是教你识别几种最系统、最反复出现的"中式学术英语"陷阱 (为什么会出现、具体长什么样), 以及怎么高效使用 LLM/语法检查这类写作辅助工具, 同时**不让工具悄悄改写你的论证逻辑**。本专题 (paper-writing-submission) 到本讲收官。

---

## 0. 这一讲要解决的具体问题, 不是"英语不够好"

很多非母语作者把写作困难简单归因为"我的英语水平不够", 于是一门心思刷词汇、背句型。但语言学界对二语学术写作的研究早就指出: **很多反复出现的问题不是词汇量问题, 而是母语的思维/语法模式系统性地"渗透"进了英语写作**——这类问题即使词汇量再大也不会自动消失, 因为它们发生在句子结构和语篇组织的层面, 需要专门识别。本讲第 1 节讲三类这样的系统性陷阱; 第 2 节讲 LLM 时代一个新问题——写作辅助工具本该帮你把地道表达和你的论证逻辑同时保住, 但用不对方法, 反而会把你的逻辑悄悄换掉。

---

## 1. 三类系统性的中式学术英语陷阱

### 陷阱① 话题前置的语篇结构, 把关键信息埋在句尾

中文习惯先铺陈背景/让步/条件, 最后才抛出结论 (语言学上称为"话题-说明"结构); 直译成英语, 常常变成一个信息重心在句尾、读者要读到最后才知道你想说什么的长句:

> **中式直译**: "As the robustness of preference optimization methods to label noise has been studied by many researchers using various techniques such as data filtering and loss reweighting, and since these techniques all have certain limitations in high-noise regimes, in this paper we propose a new method."
> **地道英语**: "Existing techniques for noise-robust preference optimization — data filtering and loss reweighting — degrade in high-noise regimes. We propose confidence-weighted DPO to address this gap."

这不只是"啰嗦"的问题, 而是**语篇组织习惯的差异**——应用语言学里著名的 **对比修辞学 (contrastive rhetoric)** 理论 (Kaplan, "Cultural Thought Patterns in Inter-Cultural Education", *Language Learning*, 1966) 最早系统提出: 不同语言/文化背景的写作者, 语篇组织的"预设路径"不同, 英语学术写作的主流范式偏好**先亮出结论/claim, 再展开支撑**的线性结构, 而作者若不自觉套用母语的语篇习惯 (先充分交代背景再引出结论), 会让英语母语审稿人读起来觉得"重点找不到"。**需要说明的是**: Kaplan 最初的表述 (把"英语=直线, 东方语言=螺旋"简化成一种民族特质) 后来被多项重新检验的研究挑战和修正 (发现同一语言内部的写作者差异往往大于语言之间的差异)——但他指出的**语篇组织习惯是可以被识别、被有意识调整的**这个核心观察, 至今仍是二语学术写作教学的基础共识。**对你的具体启示**: 回忆 L1 讲过的"叙事先行"原则 (`how_to_write_a_paper` 技能包的灵魂)——它在**段落/句子**层面同样成立: 关键 claim 要前置, 不要让读者在你交代完所有背景之后才等到重点。

### 陷阱② 名词化堆砌, 但缺少与之搭配的限制性语言 (hedging)

英语学术写作有一个成熟的"套路": **名词化结构 (nominalization, 把动词/形容词转成抽象名词, 如 "improve" → "the improvement of") 通常会搭配限制性语言 (hedging, 如 "may", "suggest", "appear to") 一起出现**, 用来准确标注一个抽象论断的确定程度。一项针对中国 EFL (English as a Foreign Language) 学习者科技论文写作的研究 (Liu, X., "The Low Co-occurrence of Nominalization and Hedging in Scientific Papers Written by Chinese EFL Learners", *Arab World English Journal*, Vol. 12, No. 1, 2021) 发现: **中国 EFL 写作者虽然同样大量使用名词化结构 (这本身是英语学术语域的正常特征), 但名词化和限制性语言的"共现率"明显偏低**——也就是说, 名词化的抽象论断经常**没有**搭配应有的 hedging, 读起来比同等证据强度下地道学术英语更"绝对"。

具体到句子层面:

> **过度绝对 (常见中式直译)**: "The experiment proves the effectiveness of the method." (一次实验用"prove"是明显过度宣称, 呼应 L1 第 3 节讲过的"无证据的 claim 是审稿人攻击点"——这里不是没证据, 而是**用词的确定程度超出了证据实际能支撑的强度**)
> **恰当限定**: "Our results suggest that the method is effective under the tested noise conditions."

**这不是"谦虚"或"自信"的文化差异问题, 而是英语学术写作对"claim 强度必须匹配证据强度"这条规则要求更显性的词汇标记** (中文里这种精确度常常靠上下文隐含传递, 英语要求你用具体的动词/情态词把它显式写出来)。

### 陷阱③ 冠词 (a/the) 遗漏或误用

中文没有冠词系统, 这是几乎所有中国英语学习者 (包括高水平写作者) 最难彻底根除的错误类型之一, 原因很朴素: 你的母语里根本没有这个语法范畴可以类推。**这类错误审稿人通常不会因此拒稿** (他们知道这是非母语写作者的常见特征), 但整篇论文里持续出现的冠词错误, 会累积成一种"作者对细节不够仔细"的印象, 值得在定稿前**专门做一遍冠词检查** (读每一个名词短语, 问自己"这里该用 a/an/the/零冠词, 为什么"), 而不是和其他类型的错误混在一起校对时顺手扫一眼。

---

## 2. 怎么用写作辅助工具, 而不丢掉自己的论证逻辑

LLM 类写作辅助工具 (以及更早的 Grammarly 类语法检查器) 能高效解决第 1 节的很多表层问题, 但也带来一个新风险: **这些工具优化的目标是"读起来流畅/地道", 不是"和你的证据强度精确匹配"**——一篇综述 ("ChatGPT in academic writing: Maximizing its benefits and minimizing the risks", *PMC*) 明确指出, 这类工具应该被当作**辅助人类判断的补充工具, 而不是替代写作者本人的批判性思考**, 原因正是: 工具不知道你实验里的证据到底支不支撑某个特定强度的表述。

**具体风险场景**: 你写了一句克制的 "Our results may indicate that confidence weighting improves robustness", 把整段丢给 LLM 说"帮我润色得更流畅/更专业", 工具很可能把它"优化"成语气更肯定的 "Our results demonstrate that confidence weighting significantly improves robustness"——读起来确实更"学术、更有力", 但**如果你的证据其实只是一次实验、没有做显著性检验 (回忆 `experiment-design` L5/L6 的频率派/贝叶斯范式), "demonstrate...significantly" 已经是你证据链撑不住的过度宣称**——这正是陷阱②的风险, 被工具在你没注意的情况下悄悄放大了。

### 推荐的具体工作流

```
① 先用你最快、最自然的语言 (可以中英混杂, 呼应本仓库一贯的记笔记习惯) 写出
   每段的核心论证骨架和 claim ——逻辑必须由你自己确定, 不外包给工具。

② 逐句润色, 而不是整段丢给工具重写——逐句检查能让你清楚看到"这一句被换成了
   什么", 而整段重写常常在流畅化的同时不知不觉挪动了 claim 的强度或段落的
   逻辑顺序。

③ 工具给出润色后的版本, 对照你②之前写的原句, 具体问一次: 「claim 的确定
   程度变了吗?」(如"may indicate"有没有被换成"demonstrate"/"prove"这类
   更强的词) ——这本质上是把 L1 讲的 narrative_audit (claim→evidence 是否
   闭合) 应用到"润色前后"这一步, 而不只是"写作时"这一步。

④ 保留修改前的版本 (diff), 方便你和合作者/导师复核逻辑有没有被悄悄改变;
   最终提交的每一句话, 无论是不是工具润色过, 责任都在你自己 (呼应 L4 第4节
   学术诚信: 工具生成的过度宣称, 提交后依然是你的责任, 不能归咎于工具)。
```

> 一句话总结这套工作流的原则: **让工具管"这句话地不地道", 自己管"这句话对不对、强度合不合适"——这两件事永远不要交给同一次操作一起完成, 否则你无法分辨到底是语言变好了, 还是论证被悄悄改写了。**

---

## 3. 常见误区

| 误区 | 真相 |
|---|---|
| 「多背高级词汇/长难句就能提升学术英语水平」 | 第1节的三类陷阱都发生在语篇组织和证据-表述匹配层面, 词汇量再大也不会自动解决, 需要专门识别和练习 |
| 「Kaplan 的"中文写作螺旋式、英语写作直线式"是铁律, 我天生就该按这个模式写」 | 后续研究已经修正了 Kaplan 最初过于简化的民族特质表述 (语言内部的写作者差异往往大于语言之间), **可调整的语篇习惯**这个核心观察仍然有效, 但不该把它当成不可改变的宿命 |
| 「把整篇论文丢给 LLM 说"帮我改成更地道的学术英语"最省事」 | 整段/整篇重写会让你难以察觉 claim 强度或逻辑顺序被悄悄改变 (第2节); 逐句处理 + 对照检查虽然慢一点, 但能保住论证的准确性, 这是不能省的步骤 |
| 「工具生成的过度宣称不是我的错, 是工具的问题」 | 提交论文的责任始终在作者本人 (呼应 L4 学术诚信红线), 使用工具不能成为"表述失实"的免责理由 |

---

## 4. 参考

- **Glasman-Deal, H., *Science Research Writing: A Guide for Non-Native Speakers of English*, Imperial College Press** —— 面向非母语科研写作者的经典系统性指南 (以 Introduction/Methodology/Results/Discussion/Abstract 五单元组织, 强调"你已经具备大部分所需的英语, 缺的是特定的写作套路"), 本讲整体方法论取向的参考书目。
- **Liu, X., "The Low Co-occurrence of Nominalization and Hedging in Scientific Papers Written by Chinese EFL Learners", *Arab World English Journal*, Vol. 12, No. 1, 2021, pp. 401-420** —— 中国 EFL 学习者科技论文中名词化-限制性语言共现不足的实证研究, 本讲第 1 节陷阱②的出处。
- **Kaplan, R. B., "Cultural Thought Patterns in Inter-Cultural Education", *Language Learning*, 16, 1966** —— 对比修辞学理论的奠基之作, 本讲第 1 节陷阱①的出处; 本讲同时指出其后续受到的学术修正, 避免过度援引其最初的简化版结论。
- **"ChatGPT in academic writing: Maximizing its benefits and minimizing the risks", *PMC*** —— LLM 类写作辅助工具应作为人类判断的补充而非替代的综述性论证, 本讲第 2 节工作流建议的依据。
- 也见本专题 `L1-from-research-to-paper.md` 第 3 节 (`narrative_audit` 的 claim→evidence 审计) —— 本讲第 2 节把这套审计思路从"写作阶段"延伸到"工具润色前后的对照检查"这一具体子场景。

---

## 5. 本讲小结 + `paper-writing-submission` 专题收官

- **三类系统性陷阱**: ① 话题前置导致关键信息埋在句尾 (需要练习"结论先行"的英语语篇习惯) ② 名词化堆砌但缺少匹配的限制性语言 (导致 claim 强度系统性偏高) ③ 冠词遗漏/误用 (需要专门的定稿检查环节)。
- 这些不是"英语不够好", 而是母语的语篇/语法模式渗透进英语写作的**系统性**现象, 可以被识别、被针对性修正。
- **写作辅助工具的正确用法**: 自己先确定逻辑骨架 → 逐句 (而非整段/整篇) 润色 → 对照检查 claim 强度有没有被悄悄改变 → 保留修改前版本、责任始终在作者本人。

至此, `paper-writing-submission` (9.7) 十讲全部完成: L1-L4 是投稿循环的主干 (装配→投稿→评审rebuttal→录用拒稿后), L5-L9 补上了主干之外、同样贯穿整个论文生命周期的现实议题 (负结果/多篇组合策略/venue深层选择/camera-ready与artifact/长期影响力), L10 收口到写作本身这个最基础的现实约束。回忆本专题 README 里的位置图: 论文写完、投出去、经历评审, 拿到录用之后, **下一站是 `research-presentation` (9.8) —— 把这项工作讲给人听。**

**动手**: 找一段你自己写过 (或正在写) 的英文摘要/引言, 逐句检查: 有没有陷阱①的话题前置结构? 每个名词化的抽象论断, 有没有搭配恰当的 hedging (陷阱②)? 如果你用过 LLM 润色, 对照润色前后, 找出至少一处 claim 强度被悄悄改变的地方 (哪怕很轻微)。
