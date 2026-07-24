# 科研写作深挖 —— 路线图与进度表

> 目标:8 个分类,由浅入深,讲透"论文这件事从叙事结构到句子层面到审稿对抗,应该怎么判断",收尾 1 篇
> [模拟审稿+rebuttal攻防 capstone](09-mock-review-rebuttal-capstone.md)。
> 定位:仓库"论文发表系列(从写作到社区回响)"5 个新系列之一(其余 4 个:
> `research-figures-deep-dive/`/`paper-submission-deep-dive/`/`academic-presentation-deep-dive/`/
> `research-release-deep-dive/`),设计文档见
> [docs/superpowers/specs/2026-07-25-paper-publication-series-design.md](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md)。

---

## 为什么不是"深挖系列"的七步模板,也不是"操作类系列"的六步模板

这是仓库第一次系统性写"文字论证判断力"类内容——没有函数签名可以"是什么",也不是一步步操作就能出结果的
工具使用(不像 SSH/Git/LaTeX 那样"跟着做就有确定输出")。判断一段摘要写得好不好、一次 rebuttal 有没有
说服力,本质上是**判断力问题**,不是"这个 API 怎么调"。因此这里设计一套全新的**六步写作判断力模板**,
每个知识点固定这个结构:

1. **常见误区/反面例子**——真实典型的差写法(不是稻草人,是这类写作里反复出现的真实模式)
2. **逐处修改对照**——同一段内容改好后的版本,逐处标注为什么改
3. **可操作检查清单**——读者下次自己写时能直接照着套用的具体规则,不是抽象原则
4. **量化验证**——能量化的部分写真实 Python 代码验证(被动语态占比、句长分布、套话密度……),
   在 `.venv` 里真实跑出结果;写作判断力本身不可 `assert`(比如"这段摘要是否吸引人"),但**支撑判断的
   统计信号可以**——代码验证的是信号,不是判断本身,这条边界每处都会明确说清楚,不混为一谈
5. **审稿人会怎么挑刺 + 反驳链**——复用 [dsa-deep-dive 20 类](../dsa-deep-dive/20-advanced-interview-depth.md)
   已验证的"五轴追问链"方法论(规模递增轴/工程约束递增轴/方案批判迭代轴/决策依据追问轴/真实性验证轴),
   这里是**评审对抗版本**:审稿人挑刺 → 作者反驳/让步 → 审稿人再追问,同样"挑 1~2 条最自然的轴线走
   2~3 层深,不强行凑满 5 轴"
6. **常见坑**

## 五轴追问链怎么适配成"评审对抗"版本

| 轴线 | 面试原意(dsa-deep-dive 20 类) | 这里的写作/评审改写 |
|------|------------------------------|---------------------|
| 规模递增轴 | 内存放得下→放不下→分布式 | 这个结论在小规模验证(如合成环境)之外,换成更大/更真实的场景还成立吗 |
| 工程约束递增轴 | 单机→并发→分布式 | 这条写作规则在字数限制/多作者协作/双盲匿名化约束下还站得住吗 |
| 方案批判迭代轴 | 连续指出具体缺陷逼换方案 | 审稿人连续追问具体的写法缺陷,逼你重新组织论证,不是笼统说"写得不好" |
| 决策依据追问轴 | 为什么选这个不选那个 | 为什么这样写不那样写——逼问选择依据而非纠错 |
| 真实性验证轴 | 简历数字被追问到具体来源 | 论文里的数字/说法被追问到"这个结论的证据链在哪、样本量多大、有没有方差" |

## 真实调研纪律

每个分类的"过来人经验"基于真实 WebSearch 调研,不是凭训练数据里的印象泛泛而谈。已核实的一手/权威来源
包括(不逐条列全,具体引用见各分类正文):

- Simon Peyton Jones, *How to Write a Great Research Paper*(七条建议原文与幻灯片,微软研究院)
- Mensh & Kording (2017), *Ten Simple Rules for Structuring Papers*, PLOS Computational Biology
  (C-C-C 框架:Context-Content-Conclusion)
- Gopen & Swan (1990), *The Science of Scientific Writing*, American Scientist(topic position / stress
  position,老信息放句首、新信息放句尾)
- Helen Sword, *Stylish Academic Writing*(2012)与她提出的"zombie nouns"(nominalization 名词化)概念
- NeurIPS 2024/2025 Reviewer Guidelines、NeurIPS Paper Checklist Guidelines 官方文档
- ICML/ICLR/NeurIPS/CVPR 关于 LLM 写作辅助的官方政策(2023-2026 各年份对比)
- *Insights from the ICLR Peer Review and Rebuttal Process*(arXiv:2511.15462,19000+ 篇论文/74000+
  条评审的大规模统计)
- 一批公开的 rebuttal 写作实战指南(Fredo Durand、Jeffrey Bigham、Maarten Sap、Niklas Elmqvist 等人的
  公开笔记)

**诚实声明**:WebSearch 返回的内容本身是二手转述(搜索引擎摘要 + AI 归纳原文),不是逐字逐句核对的原始
PDF——这和 `research/world-model-imagination-controller/` 项目"逐字读 PDF 原文"的调研标准不同,量级也
小得多。每处引用只采用多个独立搜索结果互相印证过的内容,单一来源、无法交叉印证的细节不采用。

## 真实项目素材使用声明

案例设计参考 `research/world-model-imagination-controller/`(用户真实在研、即将投稿 ICLR 的项目:世界模型
测试时想象预算分配控制器)已读取的真实文件(`01-meeting-briefing.md`/`02-deep-gap-analysis.md`/
`04-sharpened-recommendation.md`)。用法边界严格遵守设计文档规定:**不逐字复制大段真实结论当"标准答案"**,
而是借用其真实的论证结构/场景设定给范例做灵感来源——比如"摘要电梯演讲公式"的示范摘要是围绕"想象预算
分配"这个真实问题设定重新撰写的教学文本,不是从项目文件里剪切粘贴;真实数字(如 82.0% vs 63.7% 的
task-conditioning 命中率对比、Bellman telescoping 论证)在明确标注"真实项目案例"的地方直接引用,用于
展示"这是真实科研判断力长什么样",不是虚构案例。项目组自己在 `02-deep-gap-analysis.md` 里做过的一次
真实自我纠正(意识到 pilot"发现一/二"其实是 1991 年 Value of Computation 理论的直接推论、必须改口径
重新定位核心贡献,否则会被审稿人当场识别成 reinventing the wheel)是全系列最重的一处真实案例,05/06/07
三类都会从不同角度引用这同一件事——这不是巧合,是因为这一次真实的自我纠正恰好同时示范了"诚实的局限性
自曝"“多轮修改抓出叙事错误”“审稿人视角的红旗识别”三件事。

## 验证纪律

- 每一段可运行 Python 代码,写进 markdown 前先在 `e:\Workspace\dummy\.venv\Scripts\python.exe` 里真实
  跑通,确认真的能跑出写进正文的那个结果——本系列撰写过程中这一步真实抓到过至少一个 bug(见
  [04 类](04-sentence-level-academic-english.md)"引用是脚注不是名词"知识点:句子切分正则天真地按
  `. ` 切分,会被 `et al.` 里的句点误切分,导致"引用当句子主语"检测器出现假阳性——已在正文如实记录
  修复过程,不是靠事后编故事包装成"一次性写对")。
- 全部文件写完后用 [`_verify_md.py`](_verify_md.py)(直接拷贝自
  [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py),不重新设计)逐文件独立 subprocess
  跑一遍全部 ` ```python ` 代码块,确认全部通过。
- 不能自动化验证的判断(比如"这段摘要是否足够吸引人"本身是主观判断)如实标注"这是判断力问题,不是
  可 assert 的事实",不为了凑"可运行例子"硬造一个假 assert——这条纪律是设计文档明确要求的,也是本系列
  和"深挖系列"最大的方法论差异:深挖系列几乎每个知识点都能写出精确的正确性断言,这里很多时候只能验证
  "统计信号存在且方向正确",不能验证"这样写就一定会被接收"。

## 不强行凑数字声明

8 个分类合计约 40 个知识点(每类 5 个左右),不报一个跨系列对比的"总知识点数"当卖点——延续
`numpy-deep-dive` 进阶深度追加"材料相对薄弱、诚实收敛不强行凑"的先例,以及设计文档"5 个系列不会像
dsa-deep-dive 140 个知识点那样报一个大总数"的明确要求。每类知识点数量由真实能调研到、能站得住的内容
决定,不是提前定好数字再去凑。

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13.9,和其余系列共用同一个 venv)。**不需要任何
第三方包**——本系列全部代码只用 Python 标准库(`re`/`difflib`/`statistics`/`collections`),没有网络
调用、没有 GPU 依赖,是全仓库里环境要求最简单的系列之一。代码块本身运行环境和 Windows 控制台默认代码页
不完全兼容(打印中文/特殊符号在个别终端下可能显示为乱码,这是终端编码问题不是代码错误),`_verify_md.py`
和 dsa-deep-dive 一样固定设置 `PYTHONUTF8=1` 环境变量执行每个代码块,规避这个问题。

## 交叉引用声明

- 05 类"局限性自曝"和 07 类"审稿人视角"、06 类"多轮修改"三处都引用 world-model 项目同一次真实自我
  纠正,各自角度不同(自曝写法 / 红旗识别 / 修改方法论),不是重复内容,正文会互相点名交叉引用。
- 08 类"Rebuttal 写作技巧"与后续 `paper-submission-deep-dive/` 系列的"Rebuttal 时间与协作管理"分工
  明确:这里管"怎么写",那边管"怎么组织这件事"(word limit 分配、时间线、和合作者分工)——写这篇时
  `paper-submission-deep-dive/` 尚未完成,交叉引用留空,后续由主线程统一核实编号后补链接。
- 09 类 capstone 的模拟审稿意见基于 07 类调研到的真实红旗模式设计,rebuttal 回应应用 08 类技巧,是
  两类内容的应用出口,不是又一个独立分类。

## 进度表

| # | 分类 | 文件 | 知识点数 | 代码块数 | 状态 |
|---|------|------|---------|---------|------|
| 01 | 论文整体叙事结构 | [01-narrative-structure-and-elevator-pitch.md](01-narrative-structure-and-elevator-pitch.md) | 5 | 2 | ✅ 已完成(已验证) |
| 02 | Introduction 段落级写作与 Related Work 定位 | [02-introduction-and-related-work.md](02-introduction-and-related-work.md) | 5 | 2 | ✅ 已完成(已验证) |
| 03 | Method/实验部分的呈现逻辑 | [03-method-and-results-presentation.md](03-method-and-results-presentation.md) | 5 | 4 | ✅ 已完成(已验证) |
| 04 | 句子层面的学术英语写作 | [04-sentence-level-academic-english.md](04-sentence-level-academic-english.md) | 5 | 5 | ✅ 已完成(已验证) |
| 05 | 局限性/风险自曝的诚实写法 | [05-limitations-and-honest-disclosure.md](05-limitations-and-honest-disclosure.md) | 5 | 3 | ✅ 已完成(已验证) |
| 06 | 多轮修改方法论与 AI 辅助边界 | [06-revision-methodology-and-ai-boundary.md](06-revision-methodology-and-ai-boundary.md) | 5 | 3 | ✅ 已完成(已验证) |
| 07 | 审稿人视角精读——常见拒稿理由拆解 | [07-reviewer-perspective-and-rejection-patterns.md](07-reviewer-perspective-and-rejection-patterns.md) | 5 | 4 | ✅ 已完成(已验证) |
| 08 | Rebuttal 写作技巧 | [08-rebuttal-writing-techniques.md](08-rebuttal-writing-techniques.md) | 5 | 3 | ✅ 已完成(已验证) |
| 09 | 收尾:模拟审稿意见 + rebuttal 攻防 capstone | [09-mock-review-rebuttal-capstone.md](09-mock-review-rebuttal-capstone.md) | 1 篇(不计入合计) | 1 | ✅ 已完成(已验证) |

**合计:精确 40 个知识点(8 个分类文件各 5 个,逐文件 `grep -cE "^## [0-9]+\." ` 核对过,不是估算)+
1 篇 capstone,全部完成并验证。** 全系列合计 27 个 Python 代码块,用 [`_verify_md.py`](_verify_md.py)
逐文件独立 subprocess 跑过(`e:\Workspace\dummy\.venv\Scripts\python.exe -X utf8 _verify_md.py <file>`,
父进程也带 `PYTHONUTF8=1` 环境变量,避免 Windows 控制台默认代码页在捕获子进程输出时和 UTF-8 内容
冲突导致的乱码噪声——这个乱码本身不影响 `_verify_md.py` 的通过判定逻辑,判定只看子进程 `returncode`,
和输出文本能不能被正确解码显示是两回事,但父进程也带上这个环境变量能让人工复核输出时不被无关噪声
干扰),9 个文件全部 `ALL N blocks passed independently`。

**撰写过程中真实撞到的坑(如实记录,不是事后包装的叙事)**:
1. [04 类](04-sentence-level-academic-english.md)知识点 3(引用是脚注不是名词)——第一版句子切分
   正则天真地按 `. ` 切分,被 `Hafner et al. [12]` 里 `al.` 后面的句点误切分,导致"引用当句子主语"
   检测器对写对的句子出现假阳性,已修复为保护常见学术缩写(`et al.`/`e.g.`/`i.e.`/`Fig.`/`Eq.`等)
   不被误切分。
2. [03 类](03-method-and-results-presentation.md)知识点 2(Figure 1 该放什么)——检测"标题是否包含
   量化结果数字"时最初用裸 `\d`,被"Figure 1:"这个图号前缀本身的数字"1"污染,导致纯流程图标题被
   误判成"已经包含量化结果",已修复为只认百分比/小数这类更贴近"真实指标"形状的数字模式。
3. [07 类](07-reviewer-perspective-and-rejection-patterns.md)知识点 4(reinventing the wheel 红旗
   检测)——第一版关键词重叠检测直接用精确字符串匹配,"computation"(单数)和"computations"(文献
   标题里的复数)被当成两个不同的词,导致本该命中的重叠检测漏检,已修复为加入极简的单复数归并规则。
   这是全系列第三次撞见"看起来一样的意思,不是完全一样的字符串"这类文本处理教训,三次分别发生在
   句子切分、数字提取、关键词匹配三个不同的具体场景,足以说明这不是偶然的个别失误,而是用正则/规则
   处理自然语言文本时具有代表性的一类系统性陷阱。
4. [07 类](07-reviewer-perspective-and-rejection-patterns.md)知识点 4 的 capstone 应用场景([09 类](09-mock-review-rebuttal-capstone.md))里,同一个关键词重叠工具对模拟论文的文献库跑出了
   **两条**命中而不是预想的一条——除了真正的"reinventing the wheel"来源,论文的最近邻竞品工作
   (AVIC)也被列为高重叠条目。没有把这个结果当成需要消除的噪声去调参数掩盖,而是如实保留并在正文
   里说明:这恰好提醒了工具"分不清同一个定理和一个应该被比较的强相关竞品"这个真实局限,而且这条
   额外命中恰好和 Reviewer 2 独立提出的"缺 AVIC 对比"意见吻合,不是巧合而是同一件事的两个侧面。

## 不可自动化验证的判断力知识点清单(诚实边界声明)

以下知识点在正文里明确标注"这是判断力问题,不提供假验证",不是遗漏,是设计文档要求的诚实边界:
[01 类](01-narrative-structure-and-elevator-pitch.md)知识点 2(SPJ"先写论文再做研究"的哲学)、
知识点 5 的"标题吸不吸引人"判断本身;[02 类](02-introduction-and-related-work.md)知识点 3
(Related Work 放在哪个位置);[05 类](05-limitations-and-honest-disclosure.md)知识点 2(官方政策
事实,非本系列可验证的主观建议)、知识点 5(一次真实的自我纠正过程本身);
[06 类](06-revision-methodology-and-ai-boundary.md)知识点 3(找圈外人读的价值);
[08 类](08-rebuttal-writing-techniques.md)知识点 1(ICLR 真实统计数据,引用而非现场计算)。其余
知识点均提供了真实可运行的量化验证代码,验证的是"支撑判断的统计信号",不是"判断本身"——这条边界
在每处对应正文都有显式说明,不含糊带过。

---

*创建:2026-07-25,同日完成全部内容撰写与验证。*
