# 04 · 架构图与流程图深挖(Architecture & Flow Diagrams)

> 总览见 [00-roadmap.md](00-roadmap.md)。前三篇讲的都是"数据图"(有真实数值支撑的图表),这一篇讲
> "示意图"——方法框架图、pipeline 流程图、系统架构图这类**没有具体数值、靠形状和箭头讲清楚一个流程/
> 结构**的图。这类图的核心判断不是"配色对不对",是"用什么工具画"——本篇不要求你学会 TikZ 语法本身
> (那是排版工具书的范畴),只讲清楚"什么场景该选哪个工具"这个判断本身,这条判断可以被写成代码。

**本篇统一结构(七步):** 签名/是什么 → 一句话 → 底层机制(这里是工具的设计取舍)→ AI 研究场景 → 可
运行例子 → 审稿人/读者会怎么挑刺 → 常见坑。

**只有 3 个知识点,不强行凑到 4 个**:这类内容没有"可以用 matplotlib 画出来对比"的抓手,3 个知识点
分别对应 TikZ/draw.io/PowerPoint 三个工具各自的核心适用场景,加一个综合判断函数,讲透了就不再拆分——
理由见 [00-roadmap.md](00-roadmap.md) 进度表下的说明。

---

## 1. TikZ——LaTeX 原生矢量绘图的适用场景与代价

**是什么:**
```
TikZ/PGF: LaTeX 生态里的绘图宏包,用纯文本代码(而不是鼠标拖拽)描述图形
适用: 需要和正文字体/配色精确一致、需要被 git 版本控制、会被多次修订的图
代价: 学习曲线陡峭,画一个稍复杂的图往往需要反复编译-预览-调整
```

**一句话:** TikZ 的核心优势不是"画出来好看",是"图形源码是纯文本"——这意味着它能和论文正文共享同一套
字体渲染引擎(不会出现"图里的字体和正文字体不一样"这种排版事故),也能被 `git diff` 精确追踪每一次
改动,这两条对"会被多次修订、需要长期维护"的图格外重要。

**底层机制/为什么这样设计:** TikZ 图形本质是一段 LaTeX 宏代码,编译时和论文正文用同一个 TeX 引擎、
同一份字体配置渲染——这从根本上排除了"图里公式用的字体和正文公式字体不一致"这类排版瑕疵(用外部工具
画的图,字体渲染依赖那个工具自己的字体引擎,经常和 LaTeX 默认的 Computer Modern/新罗马字体不一致)。
更关键的优势在"文本源码"这个性质本身:一个几十行的 TikZ 图形描述,一次只改一个颜色属性,`git diff`
能精确高亮出改动的那一行,而不是像图片格式的二进制文件那样"整个文件被替换,看不出具体改了什么"——
下面可运行例子会现场演示这个差异。代价同样来自"纯代码描述图形"这个设计:画一个稍复杂的图,需要反复
"改代码 → 编译 → 看渲染效果 → 再改代码",这个迭代循环比鼠标拖拽慢得多,尤其是布局还没定型的早期阶段。

**AI 研究场景:** 论文里长期存在、会被反复修订的核心方法框架图(比如"controller 怎么在决策时机、
候选评估、预算调度三个环节之间交互"这类图,会从投稿版本一路改到 camera-ready 版本,中间可能经历
无数轮"审稿人说这里看不懂,再调整"),适合用 TikZ——每一轮修订都能被版本控制清楚追踪,和论文其余
`.tex` 源文件用同一套工具链管理,不需要额外维护一份图片资产文件。

**可运行例子:**
```python
import difflib

tikz_v1 = r"""
\begin{tikzpicture}
\node[draw, fill=blue!20] (a) {World Model};
\node[draw, fill=blue!20, right=of a] (b) {Controller};
\draw[->] (a) -- (b);
\end{tikzpicture}
""".strip().splitlines()

tikz_v2 = r"""
\begin{tikzpicture}
\node[draw, fill=orange!20] (a) {World Model};
\node[draw, fill=blue!20, right=of a] (b) {Controller};
\draw[->] (a) -- (b);
\end{tikzpicture}
""".strip().splitlines()

diff = list(difflib.unified_diff(tikz_v1, tikz_v2, lineterm=""))
changed_lines = [l for l in diff if (l.startswith("+") and not l.startswith("+++"))
                 or (l.startswith("-") and not l.startswith("---"))]
print("\n".join(diff))

# 只改了一个节点的填充色,git diff 应该精确定位到那一行改动,而不是把整份图当成"全部重画"
assert len(changed_lines) == 2   # 恰好一行删除 + 一行新增
assert "orange" in changed_lines[1] and "blue!20" in changed_lines[0]
print("TikZ source is plain text: a single attribute change is precisely git-diff-able")
```

**审稿人/读者会怎么挑刺:**
- "这张架构图里的字体和正文明显不是同一种字体。"——用非 LaTeX 原生工具画图时最容易被挑出来的细节,
  尤其是审稿人截图放大对比时。
- "从投稿版到 camera-ready,这张图改了什么?能不能给我看一下具体差异?"——如果整个流程用图片资产
  管理(每次修订整张图片被替换),没有人能精确回答"具体改了什么",这在需要向合作者/导师汇报修改内容
  时是实打实的沟通成本。
- "这张图的排版花了你多久?会不会在临近截止日期时因为微调一个箭头方向耗费不成比例的时间?"——TikZ
  的迭代速度劣势,在时间紧张的投稿冲刺阶段是真实的风险,不是理论上的担忧。

**常见坑:**
- 在项目早期、图的整体布局还没定型时就直接上手写 TikZ——这时候试错成本最高,布局频繁大改,更适合
  用下面第 2 条的 draw.io 先把结构定下来,布局稳定之后再考虑要不要迁移到 TikZ 做最终定版。
- 高估自己短时间内掌握 TikZ 语法的速度,在截止日期前几天才第一次尝试——这类工具值得投入的前提是
  "会被反复使用/长期维护",临时抱佛脚学一遍语法只为了画一张图,时间成本经常划不来。
- 忘记 TikZ 图形同样要遵守 [03 号文件](03-multi-panel-layout-engineering.md)的字体/线宽物理约束——
  TikZ 默认字号是相对 LaTeX 文档本身的字号设置的,同样存在"图形整体缩放导致字体比例失调"的风险,不是
  用了 TikZ 就自动免疫这类问题。

---

## 2. draw.io——GUI 快速绘图的适用场景

**是什么:**
```
draw.io(现更名 diagrams.net): 免费、跨平台、拖拽式的图形编辑器
适用: 早期头脑风暴/结构还没定型、需要非技术背景的合作者一起编辑
代价: 不和 LaTeX 字体/配色系统原生集成,版本控制只能追踪整个文件而非具体改动
```

**一句话:** draw.io 用鼠标拖拽代替写代码,单次编辑的时间成本远低于 TikZ,换来的是"结构还在反复推倒
重来的早期阶段"更快的迭代速度,代价是牺牲了文本源码带来的精确版本追踪能力。

**底层机制/为什么这样设计:** GUI 编辑器和代码化绘图工具的核心权衡是"单次编辑成本"vs"长期维护/追踪
成本"的取舍。用一个简化的工时模型把这个权衡量化:总工作量 ≈ 初始上手成本 + 修订次数 × 单次编辑成本。
draw.io 的初始上手成本几乎为零(打开就能拖框画箭头),单次编辑成本也很低(拖动一下鼠标);TikZ 的初始
上手成本高(要学语法),单次编辑成本也更高(要改代码、重新编译、检查渲染效果)。这意味着当预计修订
次数很多、但每次修订都很琐碎(挪个框的位置、改个箭头方向)时,GUI 工具的总工作量更低——这正是论文写作
早期"方法论框架还没定下来,画个草图讨论"这个场景的真实情况。

**AI 研究场景:** 和导师、合作者开会讨论方法论框架时,现场用 draw.io 拖拽几个框和箭头,比现场写 TikZ
代码快得多——[真实项目](../../research/world-model-imagination-controller/01-meeting-briefing.md)
里"该不该生成 → 该不该采纳 → 什么时候停"这类三个决策点的框架图,在思路还没完全定型、需要反复推倒重画
的讨论阶段,正是这类场景的典型例子。draw.io 支持导出 SVG/PDF 矢量格式(呼应
[03 号文件第 3 条](03-multi-panel-layout-engineering.md)),布局定型后依然能拿到矢量图的清晰度优势,
不是只能导出模糊的位图。

**可运行例子:**
```python
def estimate_total_effort(n_revisions, edit_cost_per_revision, setup_cost):
    """简化的工时模型:总工作量 = 初始上手成本 + 修订次数 x 单次编辑成本。
    不是精确的时间预测,是用来对比"这类场景该选哪个工具"的相对判断依据。"""
    return setup_cost + n_revisions * edit_cost_per_revision

# 场景A: 头脑风暴阶段,预计要改 15 次,每次都只是挪个框/改个箭头方向
gui_effort = estimate_total_effort(n_revisions=15, edit_cost_per_revision=1, setup_cost=2)
tikz_effort = estimate_total_effort(n_revisions=15, edit_cost_per_revision=4, setup_cost=15)
assert gui_effort < tikz_effort
print("brainstorm phase (15 revisions expected): draw.io total effort", gui_effort, "< TikZ total effort", tikz_effort)

# 场景B: 论文终稿阶段,预计只改 2 次(camera-ready 微调)
gui_effort_final = estimate_total_effort(n_revisions=2, edit_cost_per_revision=1, setup_cost=2)
tikz_effort_final = estimate_total_effort(n_revisions=2, edit_cost_per_revision=4, setup_cost=15)
print("camera-ready phase (2 revisions expected): draw.io total effort", gui_effort_final, "/ TikZ total effort", tikz_effort_final)
# 注意:这个简化工时模型在场景B下数字依然是 draw.io 更低,但真实决策不能唯工时论——
# "是否需要被git精确追踪""是否需要和正文字体严格一致"这些非工时因素,在终稿阶段权重更高,
# 第3条的综合判断函数会把这些因素一起纳入,不能只看这一个简化模型的输出
```

**审稿人/读者会怎么挑刺:**
- "这张图和正文的字体、配色感觉是两套体系,拼在一起有点违和。"——draw.io 导出的图形默认字体/配色
  和 LaTeX 文档不是同一套系统,需要手动调整才能贴近论文整体视觉风格,不像 TikZ 那样天然一致。
- "这张架构图上次讨论后到底改了哪里?"——GUI 工具的项目文件通常是二进制或压缩格式,版本控制系统
  只能告诉你"文件变了",不能像纯文本那样精确定位到具体修改了哪个元素。

**常见坑:**
- 早期用 draw.io 快速定型的图,到了投稿终稿阶段忘记检查字体/配色是否和论文正文视觉统一——
  "结构定下来了"和"最终排版达标"是两件不同的事,不能因为结构已经稳定就跳过最后的视觉校准。
- 团队协作时多人各自维护自己的 draw.io 副本,没有统一的共享文件/云端协作机制,导致"最新版本到底是哪
  一份"这种低级但常见的协作混乱。
- 用 draw.io 画的图直接导出成低分辨率位图贴进论文——draw.io 支持导出矢量格式,应该按
  [03 号文件第 3 条](03-multi-panel-layout-engineering.md)的判断优先导出 SVG/PDF,而不是图方便随手
  截个屏。

---

## 3. PowerPoint/Keynote 与三者的综合选择判断框架

**是什么:**
```
choose_diagram_tool(needs_latex_font_match, will_be_revised_many_times,
                     has_nontechnical_collaborator, need_it_in_next_hour) -> str
```

**一句话:** PowerPoint/Keynote 不是专业绘图工具,但胜在几乎人人都会用、打开即画,适合"临时需要一张
简单示意图,不追求长期维护"的场景;三个工具最终的选择不是"哪个更好",是一个可以显式写成判断规则的
多因素决策。

**底层机制/为什么这样设计:** 现实中研究者画图的工具选择高度碎片化——一项关于神经网络示意图绘制习惯
的调研发现,12 位受访者一共报告了 16 种不同的工具,选择工具的主要考量是"用起来顺不顺手"和"能不能
导出需要的文件格式",Inkscape/Google 画图工具/draw.io/TikZ 是最常见的几种,PowerPoint 在专业绘图
调研里占比不高,但在"需要一张示意图撑起明天的组会/slides"这类临场场景里依然是最常见的选择之一,因为
几乎不需要学习成本、团队每个人都能立刻打开编辑。这条设计判断的核心不是神化任何一个工具,是承认"画图
工具选择"本身是一个多因素权衡问题——需要多快拿到图、图要维护多久、协作者是否有技术背景、对最终视觉
一致性的要求有多高,这四个因素基本能覆盖大多数真实决策场景,可以被写成一个显式的判断函数。

**AI 研究场景:** 组会前一小时需要一张示意图讲清楚今天要讨论的想法,PowerPoint/Keynote 几乎是唯一
来得及的选择;这张图如果后来被采纳、需要放进论文正式定稿,再按上面两条的判断决定要不要迁移到 draw.io
定型结构、或者进一步迁移到 TikZ 做最终版本——工具选择不是一次性决定,同一张图在生命周期的不同阶段
可能经历"PowerPoint 草图 → draw.io 定型 → TikZ 终版"这样的迁移路径。

**可运行例子:**
```python
def choose_diagram_tool(needs_latex_font_match, will_be_revised_many_times,
                         has_nontechnical_collaborator, need_it_in_next_hour):
    """综合判断规则,按优先级从高到低依次检查:
    1. 一小时内就要用,且不要求和 LaTeX 字体精确一致 -> 用最快的工具
    2. 需要长期维护、且要求和正文字体精确一致 -> TikZ
    3. 有非技术背景的协作者要一起编辑 -> draw.io(GUI,门槛最低)
    4. 默认情况 -> draw.io(在"够用"和"上手成本"之间最平衡的默认选择)
    """
    if need_it_in_next_hour and not needs_latex_font_match:
        return "PowerPoint/Keynote"
    if needs_latex_font_match and will_be_revised_many_times:
        return "TikZ"
    if has_nontechnical_collaborator:
        return "draw.io"
    return "draw.io"

# 场景1: 论文核心方法图,会被反复修订到 camera-ready,要求和正文字体精确一致 -> TikZ
assert choose_diagram_tool(needs_latex_font_match=True, will_be_revised_many_times=True,
                            has_nontechnical_collaborator=False, need_it_in_next_hour=False) == "TikZ"

# 场景2: 和非技术背景的合作者一起画一张流程图讨论 -> draw.io
assert choose_diagram_tool(needs_latex_font_match=False, will_be_revised_many_times=False,
                            has_nontechnical_collaborator=True, need_it_in_next_hour=False) == "draw.io"

# 场景3: 一小时后就要用在组会 slides 上,不追求长期维护 -> PowerPoint/Keynote
assert choose_diagram_tool(needs_latex_font_match=False, will_be_revised_many_times=False,
                            has_nontechnical_collaborator=False, need_it_in_next_hour=True) == "PowerPoint/Keynote"

print("all three tool-choice scenarios matched expectations")
```

**审稿人/读者会怎么挑刺:**
- "这张图看起来是从 slides 里直接截的图,分辨率不太够。"——PowerPoint 草图直接被沿用到论文终稿而
  没有重新制作,是审稿阶段常见的"图片质量不达标"投诉来源,PowerPoint 适合临时/草图阶段,不适合直接
  作为投稿终稿的产出物。
- "三个工具你们是怎么分工使用的?"——如果一个团队里不同成员各自用不同工具画同一篇论文里的不同图,
  容易导致全篇视觉风格不统一,这条追问关心的是团队协作层面的一致性,不是单张图本身的质量。

**常见坑:**
- 把 PowerPoint 草图直接当终稿提交,没有意识到"够用来讨论"和"够格印在论文里"是两个不同的质量门槛。
  PowerPoint 导出的图片默认经常是较低分辨率的位图,直接贴进论文容易触发
  [03 号文件第 3 条](03-multi-panel-layout-engineering.md)讲过的矢量/位图选择问题。
- 三个工具的判断函数只是一个起点,过度机械地套用而不考虑团队自身的技术栈/协作习惯——比如团队里没人
  熟悉 TikZ,即使某张图理论上"最适合"用 TikZ,现实中依然可能应该选择团队更熟悉的工具,判断框架提供
  的是决策依据,不是不可违背的规则。
- 忽视工具迁移路径本身也有成本——"先用 PowerPoint 画草图,后面再迁移到 TikZ"这条路径,意味着最终
  还要重新绘制一遍,如果一开始就能判断出这张图注定要长期维护,直接用 TikZ 起步反而可能更省总工作量。

---

*创建:2026-07-25*
