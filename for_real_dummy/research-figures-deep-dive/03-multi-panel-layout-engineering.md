# 03 · 多面板排版工程深挖(Multi-Panel Layout Engineering)

> 总览见 [00-roadmap.md](00-roadmap.md)。前两篇([01](01-chart-type-selection.md)选类型、
> [02](02-color-and-perception.md)选颜色)讲的是单张图内部的判断,这一篇讲"好几张图/好几个子图怎么
> 拼成一张投稿级别的复合图"——这是工程活,核心矛盾是"屏幕上舒服的尺寸"和"最终印在论文里的物理尺寸"
> 经常不是一回事,本篇会用真代码把这个矛盾具体量化出来。

**本篇统一结构(七步):** 签名/是什么 → 一句话 → 底层机制(这里是印刷物理约束)→ AI 研究场景 → 可运行
例子 → 审稿人/读者会怎么挑刺 → 常见坑。

---

## 1. matplotlib `gridspec`——多面板网格怎么声明、跨行跨列

**是什么:**
```
fig.add_gridspec(nrows, ncols, width_ratios=[...], height_ratios=[...]) -> GridSpec
fig.add_subplot(gs[row_slice, col_slice]) -> Axes   # numpy 风格切片,可以跨行跨列
```

**一句话:** `gridspec` 把"这张复合图分成几行几列、每个子图占哪一块、各行各列宽窄比例多少"这几件事,
从"试出来的手工微调"变成一段声明式代码——比逐个 `plt.subplot(2,3,i)` 手工摆位置更灵活,尤其是子图
需要跨行跨列、或者宽度不相等时。

**底层机制/为什么这样设计:** `GridSpec(nrows, ncols, ...)` 声明一个虚拟网格,`width_ratios`/
`height_ratios` 控制每行每列相对宽高比例(官方文档原话:"绝对数值没有意义,只有相对比例参与计算")——
这意味着"第一个面板比第二个面板宽一倍"这种排版意图,可以直接写成 `width_ratios=[2, 1]`,不需要手工
换算成英寸。子图跨行跨列用 numpy 切片语法完成(`gs[0, :]` 表示第一行的全部列,`gs[1:, -1]` 表示从
第二行到最后一行、最后一列),这个设计选择让"网格布局"和"numpy 数组索引"共用同一套心智模型,不需要
学一套新的坐标系统。matplotlib 3.6 以后推荐用 `Figure(layout="constrained")`(或者调用
`fig.add_gridspec` 而不是独立的 `gridspec.GridSpec` 再手动摆放)替代旧的 `tight_layout()`——旧方式
在含有跨行跨列子图、colorbar 等复杂布局时经常报 "Axes that are not compatible with tight_layout"
警告(下面可运行例子如果去掉 `layout="constrained"` 会现场复现这条警告),`constrained` 布局引擎是
专门为解决这类复杂布局挤压问题设计的更新方案。

**AI 研究场景:** 论文的"核心结果图"([07 号教程体文件](07-build-a-mini-publication-figure.md)会完整
实践一遍)几乎总是多面板的——一个面板放主要的性能曲线,旁边配一个更小的面板放消融/汇总对比,`gridspec`
的 `width_ratios` 正是用来表达"主图应该比旁边的汇总图占更多版面"这种编辑判断的工具。

**可运行例子:**
```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"blue": "#0072B2", "black": "#000000"}

budget = np.array([1, 2, 3, 5, 8])
success = np.array([0.52, 0.58, 0.66, 0.74, 0.80])
rng = np.random.default_rng(3)
grid_data = rng.uniform(0.3, 0.85, size=(5, 5))

# layout="constrained": matplotlib 3.6+ 推荐的自动布局引擎,专门处理跨面板/colorbar 挤压问题
fig = plt.figure(figsize=(8, 3.2), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1.2])  # 前两个面板各占2份宽度,第三个只占1.2份
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axA.plot(budget, success, color=OKABE_ITO["blue"], marker="o")
axB.imshow(grid_data, cmap="viridis")
axC.bar(["base", "ours"], [0.6, 0.8], color=[OKABE_ITO["black"], OKABE_ITO["blue"]])
for a, letter in zip([axA, axB, axC], "ABC"):
    a.set_title(letter, loc="left", fontweight="bold")

out = os.path.join(ASSETS, "03_gridspec_demo.png")
fig.savefig(out, dpi=200)
plt.close(fig)

img = Image.open(out)
assert img.size == (round(8 * 200), round(3.2 * 200))
assert os.path.exists(out) and os.path.getsize(out) > 0
print("gridspec demo saved:", out, "size:", img.size)
```

**审稿人/读者会怎么挑刺:**
- "面板 A/B/C 的字体大小看起来不完全一致,是不是三个子图分别用不同代码生成后拼接的?"——用 `gridspec`
  在同一个 `Figure` 里统一生成的多面板图,字体/线宽天然一致;如果是用外部工具(比如 PowerPoint)把
  三张独立导出的图片拼在一起,字体不一致是最容易露馅的信号,[07 号文件](07-build-a-mini-publication-figure.md)
  会强调这一点。
- "第三个面板明显比前两个窄很多,这是故意的排版决定还是没调好?"——`width_ratios` 这类比例设置如果
  没有在图注或正文里体现出"为什么这样分配版面"的编辑意图,容易被解读成排版能力不足而不是有意为之。
- "把这几个面板换成同样大小的独立小图分开放,是不是比硬挤在一起更清楚?"——多面板不是无脑越多越好,
  面板数量多到每个面板本身太小、看不清坐标轴标签时,拆开反而更清楚。

**常见坑:**
- 忘记用 `layout="constrained"`(或者旧版本的 `fig.tight_layout()`)会导致面板之间标签互相重叠、
  或者子图被推出画布边界——这个坑在只有一个面板时不会出现,只有多面板+跨行跨列+colorbar 组合到一起
  时才会冒出来,容易在开发阶段被忽略,直到真正拼多面板图才发现。
- `width_ratios`/`height_ratios` 的数值只有**相对**比例参与计算(官方文档原话),`[2, 2, 1.2]` 和
  `[20, 20, 12]` 效果完全一样——容易误以为这些数字有绝对单位意义,费力去凑"精确的英寸数"。
- 跨行跨列切片(`gs[1:, -1]`)的切片语法和 numpy 数组切片规则完全一致,但索引的是"网格里的格子"不是
  "像素",容易和后面英寸/像素相关的计算(下面第 2-4 条)混为一谈。

---

## 2. 字体/线宽/DPI 三件套——排版参数背后的物理约束

**是什么:**
```
matplotlib.rcParams["font.size"]      # 默认 10.0 (pt)
matplotlib.rcParams["axes.linewidth"] # 默认 0.8 (pt),坐标轴边框线宽
matplotlib.rcParams["lines.linewidth"]# 默认 1.5 (pt),数据线默认线宽
```

**一句话:** 字体大小、线宽、DPI 这三个参数不是互相独立调的审美选项,它们共同服务于同一个物理约束——
"这张图最终会被印/显示多大",三者必须按这个目标尺寸联动设置,不是各自拍脑袋定一个"看起来还行"的数字。

**底层机制/为什么这样设计:** matplotlib 的字体大小、线宽单位都是**点(point, pt)**,这是一个**物理
长度单位**(1 pt = 1/72 英寸),不是"屏幕像素"这种分辨率相关的单位——这意味着同一份 `font.size=10`
的代码,不管你 `figsize` 开多大、`dpi` 设多少,理论上"物理尺寸"应该是恒定的 10pt。真正决定"这段文字
占图片多大比例"的,是 `figsize`(英寸)相对于最终印刷/显示尺寸的比例,不是字体本身的 pt 数——这个
关系[下面第 4 条](#4-作者时尺寸陷阱为什么必须以最终印刷宽度-authoring)会展开算清楚。DPI(dots per
inch)则是"每英寸多少像素点"的换算系数,只对位图(PNG)导出起作用——`fig.savefig(path, dpi=300)`
产出的像素宽度精确等于 `figsize 宽度(英寸) × 300`。社区常见的发表级默认值大致是:字体 7-9pt、线宽
0.5-1.0pt(比 matplotlib 默认的 `axes.linewidth=0.8`/`lines.linewidth=1.5` 略细,因为默认值是给
"交互式屏幕查看"调的,缩小到论文单栏宽度后线条会显得过粗)、PNG 位图导出至少 300 dpi(印刷惯例)。
这些不是 matplotlib 官方文档强制规定的数字(官方文档本身没有给出"发表推荐 dpi"这类具体建议,只是把
`dpi` 参数的默认行为——跟随 `rcParams["savefig.dpi"]`,而后者默认值是字符串 `"figure"`,即跟随
`figure.dpi`——文档说清楚),而是社区教程反复沉淀下来的经验共识。

**AI 研究场景:** 会议模板通常规定单栏图片的最大宽度(比如约 3.3 英寸,双栏排版的常见单栏宽度),写
论文时最常踩的坑就是在一个 13 寸笔记本屏幕上把图调得"看起来刚刚好",而这个"刚刚好"是按屏幕上远大于
3.3 英寸的显示尺寸调出来的——[第 4 条](#4-作者时尺寸陷阱为什么必须以最终印刷宽度-authoring)会现场
演示这个换算陷阱具体会造成多大的字体缩水。

**可运行例子:**
```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO_BLUE = "#0072B2"

# 先确认 matplotlib 的默认值(实测,不是凭记忆写的数字)
assert matplotlib.rcParams["axes.linewidth"] == 0.8
assert matplotlib.rcParams["font.size"] == 10.0
assert matplotlib.rcParams["lines.linewidth"] == 1.5
print("default axes.linewidth:", matplotlib.rcParams["axes.linewidth"])
print("default font.size:", matplotlib.rcParams["font.size"])
print("default lines.linewidth:", matplotlib.rcParams["lines.linewidth"])

# 按单栏论文宽度(3.3 英寸)直接 authoring,调低字体/线宽到适合缩小后阅读的数值
fig, ax = plt.subplots(figsize=(3.3, 2.4))
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
ax.plot([0, 1, 2], [0, 1, 0.5], color=OKABE_ITO_BLUE)
ax.set_xlabel("x")
out = os.path.join(ASSETS, "03_pub_style_demo.png")
fig.savefig(out, dpi=300)
plt.close(fig)

img = Image.open(out)
expected_px = (round(3.3 * 300), round(2.4 * 300))
assert img.size == expected_px   # 像素宽高 = figsize(英寸) x dpi,精确整数关系
print("pub-style figure saved:", out, "size:", img.size, "expected:", expected_px)
```

**审稿人/读者会怎么挑刺:**
- "把这篇论文打印成纸质版,这张图里的坐标轴刻度数字还能看清楚吗?"——审稿人经常是在平板或者打印稿上
  审稿,不是在你调图时用的高分辨率大屏幕上看,字体过小是审稿阶段真实会发生的可读性事故。
- "这张图的线条比正文表格里的线粗了好几倍,两者视觉上不协调"——同一篇论文里多张图/表如果线宽/字体
  比例不统一,会显得像是几个人分别做的,拼接感很重。
- "PNG 图片是用什么 DPI 导出的?我看清晰度不太够。"——投稿系统通常有位图分辨率的下限要求,dpi 太低
  会在评审阶段被直接要求返工。

**常见坑:**
- 在远大于最终印刷尺寸的屏幕上调字体大小,凭"看起来舒服"确定字号——这是[第 4 条]
  (#4-作者时尺寸陷阱为什么必须以最终印刷宽度-authoring)要专门拆解的陷阱,本条先点出问题存在。
- 把线宽调得比默认值更粗以为"更醒目",忽略了缩小到论文单栏宽度之后线条会被压缩得比例更粗、显得笨重
  甚至糊在一起——线宽和字体一样需要按"最终印刷尺寸"反推,不是在大屏幕上凭直觉加粗。
- 同一篇论文的多张图之间字体/线宽设置不统一(有的图用了默认值,有的图手动调过)——最省心的做法是
  在脚本开头统一调用一次 `matplotlib.rcParams.update({...})`,让全篇所有图共享同一套排版参数,而不是
  每张图各自设置一遍容易漏改。

---

## 3. 矢量图(PDF/SVG)vs 位图(PNG)——什么时候必须用哪种

**是什么:**
```
fig.savefig(path, dpi=300)      # 位图:.png/.tif -- 存储的是像素网格,放大会糊
fig.savefig(path)               # 矢量图:.pdf/.svg -- 存储的是绘图指令,任意缩放不失真
```

**一句话:** 折线图/柱状图/散点图这类由线条、形状、文字组成的图,应该优先导出矢量格式(PDF/SVG)——
它们存的是"怎么画"的指令而不是像素颜色,缩放到任意尺寸都不会模糊;只有真实照片/显微图像这类连续色调
的内容,才应该用位图(PNG/TIFF)。

**底层机制/为什么这样设计:** 位图本质是一个像素颜色网格,放大到超过原始像素密度就会出现锯齿或模糊——
这也是为什么位图必须提前确定好目标 DPI(通常 ≥300)才能保证印刷质量。矢量格式存储的是数学描述(路径、
坐标、字体引用),渲染引擎在任意缩放比例下都重新计算出清晰的输出,天然不存在"分辨率不够"这个问题,
文件体积也常常比同等视觉复杂度的高分辨率位图更小(下面可运行例子会现场比较三种格式导出同一张简单折线
图的字节数)。绝大多数期刊/会议的图片提交要求明确区分这两类:线条图/示意图类要求矢量格式(常见是 PDF
或 EPS),真实照片类才允许位图,但即使是位图也通常要求 ≥300 ppi(每英寸像素数,和 DPI 是同一个概念在
不同语境下的说法)。

**AI 研究场景:** 论文正文里几乎所有的实验结果图(性能曲线、消融柱状图、热力图)都属于"应该用矢量格式"
的类别——LaTeX 编译流程里 `\includegraphics{figure.pdf}` 直接嵌入矢量图后,读者在 PDF 阅读器里无限
放大依然清晰,这对审稿人想要仔细核对某个数据点/误差棒范围时格外重要。反例是"loss landscape 的真实
渲染截图""机器人真实摄像头画面"这类连续色调图像,矢量格式对这类内容没有意义(线条/路径描述无法表达
照片里的连续色调),应该用高分辨率 PNG。

**可运行例子:**
```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO_BLUE = "#0072B2"

fig, ax = plt.subplots(figsize=(3, 2))
ax.plot([0, 1, 2], [0, 1, 0.5], color=OKABE_ITO_BLUE)
png_path = os.path.join(ASSETS, "03_export_demo.png")
pdf_path = os.path.join(ASSETS, "03_export_demo.pdf")
svg_path = os.path.join(ASSETS, "03_export_demo.svg")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
fig.savefig(svg_path)
plt.close(fig)

# 验证方式(见 00-roadmap.md):位图核对像素尺寸,矢量图核对格式签名——两者标准不同,不能混用
img = Image.open(png_path)
assert img.size == (round(3 * 300), round(2 * 300))

with open(pdf_path, "rb") as f:
    assert f.read(5) == b"%PDF-"          # PDF 文件精确的格式签名
with open(svg_path, "rb") as f:
    head = f.read(200)
    assert b"<?xml" in head               # SVG 是 XML 文本格式

png_bytes = os.path.getsize(png_path)
pdf_bytes = os.path.getsize(pdf_path)
svg_bytes = os.path.getsize(svg_path)
print("same simple line plot, file size (bytes) across three export formats:")
print("  PNG (300dpi raster):", png_bytes)
print("  PDF (vector):", pdf_bytes)
print("  SVG (vector, XML text):", svg_bytes)
# 对这种"只有几条线"的简单图,矢量格式的文件体积反而明显小于同等清晰度的位图
assert pdf_bytes < png_bytes
```

本机实测(同一张 3×2 英寸的简单折线图):PNG(300dpi)约 31KB,SVG 约 12KB,PDF 约 6KB——矢量格式
不仅缩放不失真,对这种"由少量线条构成"的图,文件体积反而比位图更小,这是很多人直觉上会搞反的一点
("矢量图应该更大更复杂"这个印象通常来自 Adobe Illustrator 那类含大量图层/滤镜的复杂设计文件,不
适用于 matplotlib 画的简单数据图)。

**审稿人/读者会怎么挑刺:**
- "把这张图放大两倍再看,坐标轴数字变模糊了,是不是提交的位图分辨率不够?"——用了低 DPI 的 PNG 是
  最直接会被抓到的问题,审稿人放大看细节(比如核对某个数据点的具体位置)是常见的审稿动作。
- "为什么示意图/架构图这类应该是矢量的内容,提交的是一张位图截图?"——[04 号文件]
  (04-architecture-and-flow-diagrams.md)会讲架构图工具的选择,但不管用什么工具画,最终导出格式
  同样要遵守"线条图用矢量"这条原则。
- "补充材料里的真实机器人摄像头画面为什么是 PDF 格式,文件反而很大还是糊的?"——反过来的错误同样会
  被挑出来:真实照片类内容用矢量格式封装,本质上还是内嵌了一张位图,既拿不到矢量的缩放优势,还可能
  导致文件体积异常膨胀。

**常见坑:**
- 期刊/会议要求矢量图内嵌的位图元素(比如图里贴了一张真实截图)分辨率同样要 ≥300 ppi 才算数——不是
  "只要整体文件是 PDF 格式就自动达标",PDF 内部嵌入的位图部分依然要单独满足分辨率要求。
- 把 PDF/SVG 矢量图直接当成"万能格式"用在需要连续色调的照片上,既拿不到矢量的优势(照片不是路径
  描述能表达的内容),又可能因为内部实际封装的仍是位图数据而导致文件体积不降反增。
- 有的旧版画图工具/PPT 导出的"矢量图"实际上是把整个画面栅格化后包了一层矢量容器(比如导出 PDF 时
  意外触发了"另存为图片"路径),不能只看文件扩展名就认为一定拿到了真正的矢量数据,最好导出后放大
  预览确认线条是否依然锐利。

---

## 4. "作者时尺寸"陷阱——为什么必须以最终印刷宽度 authoring

**是什么:**
```
effective_printed_pt = original_pt * (final_width_in / authored_width_in)
# 在比最终尺寸更大的画布上调好的字号,导出时被压缩到最终尺寸后,有效字号会等比例缩水
```

**一句话:** 在舒适的大屏幕上把 `figsize` 开到 10 英寸宽、字体调到看起来舒服的大小,最后用
`\includegraphics[width=\linewidth]` 把图缩进论文实际约 3.3 英寸宽的单栏——这个缩放过程会让"看起来
刚刚好"的字号跟着等比例缩水到不到三分之一,这是排版新手最常踩、也最容易被忽视的一个坑,因为**开发阶段
屏幕上看到的效果和最终印刷效果完全不是同一回事**。

**底层机制/为什么这样设计:** matplotlib 的 `font.size` 是相对于 `figsize`(英寸)而言的物理量——
在一张 10 英寸宽的画布上,8pt 字体只占画布宽度很小的一部分;如果这张图之后被"压缩"进 3.3 英寸宽的
论文单栏(不管是 LaTeX 的 `width=\linewidth` 缩放,还是手动导出时改小 `figsize`),整张画布(包括
字体)按同一个比例整体缩小,原本"看起来还行"的 8pt 字体,有效印刷大小会缩到 `8 × 3.3/10 ≈ 2.6pt`——
远低于任何印刷惯例认为可读的下限。真正正确的做法是**从一开始就把 `figsize` 设成最终会被印刷的物理
尺寸**(比如真的是 3.3 英寸宽),在这个真实尺寸下调字体/线宽,所见即所得,不需要任何事后缩放,自然
就不会有这个陷阱。

**AI 研究场景:** 这是"论文图表在提交阶段被审稿人吐槽'字太小看不清'"最常见的技术根源,尤其是团队里
习惯了用大屏幕(甚至外接显示器)开发画图脚本的研究者,如果没有专门测试过"缩小到真实单栏宽度之后长
什么样",很容易在提交前才第一次发现问题——这时候往往已经临近截止日期,来不及从头调整。

**可运行例子:**
```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO_BLUE = "#0072B2"

def font_pixel_height(font_pt, dpi):
    """一个 font_pt 磅值的文字,在给定 dpi 下最终渲染出多少像素高——用来量化"缩水"到底缩了多少。"""
    return font_pt * dpi / 72.0

# 正确做法: 直接按最终印刷宽度(3.3寸)authoring,dpi=300 是最终真实的导出分辨率
correct_px = font_pixel_height(8, dpi=300)
# 陷阱做法: 在 10 寸画布上调好同样标称"8pt"的字体,但最终要塞进 3.3 寸,等效 dpi 只有 300*3.3/10=99
shrunk_px = font_pixel_height(8, dpi=99)

assert round(correct_px, 1) == 33.3
assert shrunk_px < correct_px / 2.5   # 缩水到不足三分之一
print("correct-authoring effective font size (px):", round(correct_px, 1))
print("trap-authoring effective font size (px):", round(shrunk_px, 1))

# 两张图都导出成同样的最终像素宽度(便于直接对比观感),分别代表"正确"和"陷阱"两种 authoring 方式
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.plot([0, 1, 2], [0, 1, 0.5], color=OKABE_ITO_BLUE)
ax.set_xlabel("authored at true final size (8pt)", fontsize=8)
out_correct = os.path.join(ASSETS, "03_authored_correct.png")
fig.savefig(out_correct, dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6.67))   # 同样的宽高比,但画布放大了约3倍
ax.plot([0, 1, 2], [0, 1, 0.5], color=OKABE_ITO_BLUE)
ax.set_xlabel("authored oversized, same 8pt spec, shrunk on save", fontsize=8)
out_shrunk = os.path.join(ASSETS, "03_authored_shrunk.png")
fig.savefig(out_shrunk, dpi=99)   # 刻意调低 dpi,让最终像素宽度和上面一致,模拟"被压缩进论文单栏"
plt.close(fig)

img_correct = Image.open(out_correct)
img_shrunk = Image.open(out_shrunk)
# 两张图的最终像素宽度做到几乎一致(可以公平对比观感),但文字/线条的相对粗细天差地别
assert abs(img_correct.size[0] - img_shrunk.size[0]) <= 1
print("correct image pixel size:", img_correct.size, "/ shrunk image pixel size:", img_shrunk.size)
```

两张图最终导出的像素宽度几乎一致(990px 左右,代表"同样被塞进了论文单栏这么宽的地方"),但用 Read
工具实际打开对比会发现:"正确" authoring 的那张图坐标轴数字清晰可读、线条粗细适中;"陷阱" authoring
的那张图坐标轴数字小到几乎看不清、线条细如发丝——这正是很多论文投稿后被审稿人吐槽"图看不清"的真实
成因,不是审稿人在挑刺,是这张图在真实印刷尺寸下确实不可读。

**审稿人/读者会怎么挑刺:**
- "你们提交的补充材料 PDF 里这张图,我把 PDF 阅读器缩放到 100% 之后完全看不清坐标轴刻度写的什么。"——
  这条批评几乎精确复现上面演示的陷阱,而且审稿人通常不会追问原因,只会记一条"figure quality poor"
  的负面印象。
- "这几张图的字体大小看起来不统一,有的清晰有的模糊,是不是有的图用大屏幕调的、有的图专门按最终尺寸
  调过?"——同一篇论文里如果部分图踩了这个陷阱、部分图没踩,不一致本身就是一个信号。

**常见坑:**
- 只在"能不能跑出图"这个层面测试代码,从来没有真正模拟过"这张图被压缩进论文单栏之后长什么样"这一步——
  修复方法很简单但容易被跳过:导出后用图片查看器把图缩放到最终印刷会呈现的实际大小看一眼,而不是只在
  开发时的大窗口里瞄一眼觉得"看起来还行"。
- 以为只要 `dpi` 设置得足够高就能规避这个问题——DPI 只影响位图的像素密度上限,不影响"字体相对画布的
  比例"这个根本关系,再高的 DPI 也救不回来一个从一开始就按错误画布尺寸调出来的字号比例。
- 团队协作时不同成员用不同尺寸的屏幕/习惯的 `figsize` 各自画图,拼进同一篇论文后各张图的"有效印刷
  字号"互相不一致——统一在脚本开头写死目标 `figsize`(对应真实的最终印刷宽度)是最简单的规避方式。

---

*创建:2026-07-25*
