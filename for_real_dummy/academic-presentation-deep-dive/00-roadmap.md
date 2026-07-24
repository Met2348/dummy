# 学术演讲与参会深挖 —— 路线图与进度表

> 目标:5 个分类 + 1 篇叙事体 capstone,讲透"论文写完、投出去之后,怎么在会议现场真正把它讲清楚"——
> Oral Presentation 怎么组织节奏、Slides 怎么设计信息密度、Poster 怎么摆摊讲解、现场 Q&A 被问住了怎么
> 接、参会本身(networking/日程/礼仪)怎么过得有价值。
> 定位:仓库"论文发表系列(从写作到社区回响)"5 个新系列之一(其余 4 个:
> [research-writing-deep-dive](../research-writing-deep-dive/00-roadmap.md)、
> [research-figures-deep-dive](../research-figures-deep-dive/00-roadmap.md)、
> [paper-submission-deep-dive](../paper-submission-deep-dive/00-roadmap.md)、
> [research-release-deep-dive](../research-release-deep-dive/00-roadmap.md)),设计文档见
> [docs/superpowers/specs/2026-07-25-paper-publication-series-design.md](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md)。

**背景**:导师 Weikai Lin(与 [worldmodel-imagination-controller-research] 项目记忆记录的同一位导师)
第二轮反馈里明确要求补充"参会时的各种指南、oral、poster、presentation"这类此前仓库完全没有覆盖过的
内容。这条系列直接响应这部分要求。

**为什么不报一个"N 个知识点"的总数**:写作/演讲类内容天然是"原则+范例"结构,不是像 numpy/dsa 那样
可以拆成原子化的函数级/算法级知识点。延续 numpy-deep-dive 进阶深度追加"材料相对薄弱、诚实收敛不强行
凑"的先例,本系列按"5 个分类"呈现,不报一个凑出来的知识点大总数。

---

## 和仓库其余内容的边界(避免重复劳动)

- **和 `research-figures-deep-dive` 的边界**:那个系列讲论文里的图表怎么做(投稿到期刊/会议,给审稿人
  反复精读用);这里讲"同样一份工作的可视化内容,搬上讲台/海报之后,设计原则要跟着换一套"——媒介从
  "印在纸上给人精读"变成"投在幕布上给人一闪而过地听",信息密度要求完全不同。两个系列不重复覆盖同一层
  内容,[02-slides-design-principles.md](02-slides-design-principles.md) 开头明确写了这条边界声明。
- **和 `research-writing-deep-dive` 的边界**:那个系列的 Rebuttal 写作技巧(08 号文件)是书面的、有
  几天到几周准备时间;本系列 [04-live-qa-skills.md](04-live-qa-skills.md) 讲的现场 Q&A 是口头的、几秒钟
  内必须张嘴——两者共享一部分底层判断力(比如"承认合理批评比硬辩专业"),但训练方式不同,04 号文件
  开头和结尾都明确做了这组对照,不是重复内容。本系列的六步模板本身也是直接复用该系列设计的"六步写作
  判断力模板",只把第⑤步替换成"听众/评委会怎么问"(详见下一节)。
- **和 `daily-toolkit-deep-dive/07-latex-paper-writing.md` 的边界**:那篇文档已经用本机 MiKTeX 真实
  编译验证过 LaTeX 论文写作的完整流程。本系列
  [02-slides-design-principles.md](02-slides-design-principles.md) 第 3 节讨论"要不要用 LaTeX Beamer
  做 slides"这个工具选择判断时,交叉引用那篇文档,不重复 LaTeX 编译细节本身。
- **和本系列自己的 capstone 素材来源**:[06-conference-day-capstone.md](06-conference-day-capstone.md)
  借用用户真实在研项目 `research/world-model-imagination-controller/` 的真实论证结构、真实数字、真实
  发生过的自我核验过程做场景设计,文件开头有明确的"这是假设性场景,不是真实已经发生的会议经历"声明,
  不冒充这是已发生的真事,也不代写用户论文本身的内容。

---

## 六步演讲判断力模板

复用 `research-writing-deep-dive` 系列为科研写作设计的"六步写作判断力模板"(定义可见该系列
[00-roadmap.md](../research-writing-deep-dive/00-roadmap.md)),把第⑤步从"审稿人会怎么挑刺"换成
"听众/评委会怎么问"——审稿人的追问隔着几周书面时间,可以反复斟酌措辞;听众的追问是几秒钟内当场发生的,
同一套"五轴追问链"方法论(规模递增轴/工程约束递增轴/方案批判迭代轴/决策依据追问轴/真实性验证轴,
源自 [dsa-deep-dive 20 类](../dsa-deep-dive/20-advanced-interview-depth.md)已验证的方法论),这里改写
成现场口语版,应对的从容程度完全不同——具体改写对照表见
[01-oral-presentation-structure-and-pacing.md](01-oral-presentation-structure-and-pacing.md) 第 1 节。

1. **常见误区/反面例子**——真实典型的差范例
2. **逐处修改对照**——同一段内容改好后的版本,逐处标注为什么改
3. **可操作检查清单**——读者下次自己准备演讲/海报时能直接照着套用的具体规则
4. **量化验证(真实代码)**——能量化的部分写真实 Python 代码验证(信息密度、语速换算的时长预算、
   字号-可读距离拟合、回应时长、留白率……),在 `.venv` 里真实跑出结果;演讲判断力本身不可 `assert`
   (比如"这场 Q&A 应对是否得体"),但**支撑判断的量化信号可以**——代码验证的是信号,不是判断本身,
   这条边界在每处用到的地方都明确说清楚,不混为一谈
5. **听众/评委会怎么问**——现场口语版追问链
6. **常见坑**

---

## 验证方式声明(图片类内容的验证颗粒度,和纯文本 assert 不同)

本系列包含仓库里少见的"真实渲染图片文件"内容(matplotlib 生成的 slide/poster 版式对照图),延续
`research-figures-deep-dive` 系列率先声明过的方法论差异:**图片的像素内容本身不能被 `assert`**,
能做、也确实做了的是:

1. **文件真实生成**——`Path(...).exists()` 真实检查。
2. **渲染时的真实测量指标**——用 matplotlib 渲染器的真实文字包围盒(`get_window_extent`)量出的
   信息块数量、总字数、字号、留白面积占比,这些是从真实渲染结果里量出来的数字,不是凭印象断言的。
3. **溢出检查**——每个文字/图形元素的包围盒是否落在画布边界内,真实捕捉过一次"标题字号太大导致
   文字排到画布外"的真实事故并修复,详见 [02-slides-design-principles.md](02-slides-design-principles.md)
   第 1 节。
4. **如实报告不完全符合直觉的测量结果**——[02-slides-design-principles.md](02-slides-design-principles.md)
   第 1 节的"信息密度"对照图,四项量化指标里有一项(文字油墨覆盖面积占比)和"坏例子更拥挤"这个直觉
   不完全一致(一个 64pt 大数字的像素覆盖面积可以超过八行小字),这个反直觉结果被原样保留、写进正文
   讨论,不是挑一个显得好看的指标回避掉。

不冒充"图片内容已经过像素级验证",这条颗粒度差异写在这里统一声明,各分类文件不重复这段declaration,
只在用到的地方简短提示"验证方式见 00 号文件声明"。

---

## 真实调研来源(跨分类共同纪律)

每篇的"常见坑"和量化规则,均基于真实 WebSearch 调研(而不是凭训练数据印象撰写),具体来源列在各分类
文件末尾的"参考来源"小节。核心来源包括:

- Simon Peyton Jones, *How to Give a Great Research Talk*(与 *How to Write a Great Research Paper*
  并列的独立姊妹讲座)——[simon.peytonjones.org/great-research-talk](https://simon.peytonjones.org/great-research-talk/)。
- Mike Morrison 发起的 `#BetterPoster` 运动与 Colin Purrington 的经典海报设计指南
  ([colinpurrington.com/tips/poster-design](https://colinpurrington.com/tips/poster-design/))。
- 多所高校海报设计指南给出的字号-可读距离对照数据(6/10/12/14 英尺 → 30/48/60/72pt)。
- 学术会议 Q&A 应对指南(CAP 回应结构、三档问题分类法、承认合理批评的专业化原则)。
- 学术会议 networking 与 poster session 参观者礼仪指南。

---

## 运行环境

仓库根目录 `e:\Workspace\dummy\.venv`(Windows 原生,Python 3.13.9,和其余系列共用同一个 venv)。本系列
用到 `matplotlib`(真实渲染 slide/poster 对照图)、`numpy`(字号-距离线性拟合)、标准库
`re`/`textwrap`——均已在 `.venv` 里确认可用(`matplotlib 3.10.9`、`PIL 12.2.0`,后者本次未直接用于
生成图片但按任务要求确认过可用性)。

**一处值得记录的真实环境坑**:本机默认区域设置下,Python 子进程的标准输出解码走的是 GBK(`cp936`)
而不是 UTF-8。`_verify_md.py` 在 `subprocess.run(...)` 里已经给**被验证的代码块**传了
`PYTHONUTF8=1`,但如果验证脚本自己(外层进程)不是在同样的 UTF-8 模式下启动,Python 的
`subprocess` 读取子进程输出的后台线程会在代码块打印中文/特殊符号时抛出
`UnicodeDecodeError`(报在一个独立的 reader 线程里,不影响最终 pass/fail 判定,但会打印一堆吓人的
troceback 噪音,容易被誤认成真失败)。**结论:在这类区域设置的机器上,运行
`python _verify_md.py <file>.md` 时,外层调用也要显式设置 `PYTHONUTF8=1`**(比如
`PYTHONUTF8=1 python _verify_md.py 01-....md`),不是只给内层子进程设置——本系列全部验证记录均已
按这个方式复核过,确认是环境层面的噪音而非代码错误。

---

## 进度表

| # | 分类 | 文件 | 代码块 | 状态 |
|---|------|------|--------|------|
| 00 | 路线图 | [00-roadmap.md](00-roadmap.md)(本文件) | — | ✅ |
| 01 | Oral Presentation 结构与讲述节奏 | [01-oral-presentation-structure-and-pacing.md](01-oral-presentation-structure-and-pacing.md) | 4/4 通过 | ✅ 已完成(已验证;含 talk 时间预算审计工具、首尾呼应关键词重合度检测、多轮排练收敛模拟) |
| 02 | Slides 设计原则 | [02-slides-design-principles.md](02-slides-design-principles.md) | 3/3 通过 | ✅ 已完成(已验证;含真实渲染的信息密度对照图 `_assets/bad_slide.png`/`good_slide.png`,如实记录"文字油墨覆盖面积"这项指标和直觉不完全一致的真实测量结果;公式密集度量化启发式工具) |
| 03 | Poster 设计与摆摊讲解 | [03-poster-design-and-pitching.md](03-poster-design-and-pitching.md) | 3/3 通过 | ✅ 已完成(已验证;含真实渲染的 poster 版面对照图 `_assets/bad_poster.png`/`good_poster.png`,海报物理尺寸设成真实的 36×48 英寸使 fontsize 单位和真实印刷字号直接对应;字号-可读距离线性拟合工具;电梯讲解长度决策工具。撰写 poster 渲染脚本时现场发现 matplotlib 的 `wrap=True` 是按整个画布宽度换行、不是按局部方块宽度换行,窄栏文字会越界写进相邻方块,已改成按真实物理宽度手动 `textwrap` 修复) |
| 04 | 现场 Q&A 应对技巧 | [04-live-qa-skills.md](04-live-qa-skills.md) | 4/4 通过 | ✅ 已完成(已验证;三档问题分类工具、CAP 回应时长量化对照、承认式语言关键词检测器。第 3 节用取材真实项目 `02-deep-gap-analysis.md` 记录的一次真实自我核验[项目组发现自己"新发现"其实是1991年经典VOC理论的推论]设计了"被问到犀利问题当场怎么接"的具体范例。撰写时现场发现第 4 节"情绪化vs技术词"检测器的测试句子里意外把"baseline"同时写进了泛泛而谈和具体化两个对照句,导致断言失败,已修正测试句子后复核通过) |
| 05 | 会议参会实用指南 | [05-conference-attendance-guide.md](05-conference-attendance-guide.md) | 3/3 通过 | ✅ 已完成(已验证;日程冲突检测工具[含"首尾相接不算冲突"边界情况]、联系人跟进优先级工具、"评论伪装成问题"检测启发式) |
| 06 | 会议日 capstone | [06-conference-day-capstone.md](06-conference-day-capstone.md) | 2/2 通过 | ✅ 已完成(已验证;叙事体,明确标注"假设性场景"声明,串联 01-05 全部方法论,直接复用 02/03 号文件已经渲染验证过的 slide/poster 图片作为"这篇假设论文的真实素材",4 轮 Q&A 追问全部取材真实项目文档的真实内容[范围声明/基线可复现性调研/VOC理论自我核验]改编,不虚构项目细节) |
| — | 验证脚本 | [_verify_md.py](_verify_md.py) | — | ✅ 从 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py) 原样拷贝(`diff` 确认字节级一致) |

**合计:5 个分类 + 1 篇叙事体 capstone,19 个 Python 代码块全部独立验证通过,4 张真实渲染的对照图片。**

---

## 撰写与验证纪律

- 每个可运行代码块必须先在仓库根目录 `.venv`(`e:\Workspace\dummy\.venv\Scripts\python.exe`)里真实
  跑通,再写进 markdown——本系列撰写过程中真实发生过两次"先跑发现不对、改完再跑通过"的情况(poster
  渲染的文字越界问题、Q&A 检测器的测试句子重叠问题),均已如实记录在上方进度表对应行,不是编出来的
  "顺利叙事"。
- 图片类内容的验证颗粒度按本文件"验证方式声明"一节执行,不冒充像素级验证。
- 不能自动化验证的判断力内容(比如"这场即兴回应是否得体""这次破冰是否自然"),各分类文件在对应
  小节如实标注"这是判断力问题,不能被 assert",不强行造一个看似客观实则牵强的断言。
- 场景设计涉及真实项目素材 `research/world-model-imagination-controller/` 的地方,均先实地读取该
  目录下的真实文件(`00-brainstorm-10-ideas.md`/`01-meeting-briefing.md`/`02-deep-gap-analysis.md`/
  `04-sharpened-recommendation.md`/`07-baseline-reproducibility-audit.md`)后再设计,不凭空编造项目
  细节;不逐字复制大段真实结论当作教学"标准答案",只借用其真实论证结构/场景设定。

---

*创建:2026-07-25*
