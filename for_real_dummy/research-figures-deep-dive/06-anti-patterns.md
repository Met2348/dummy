# 06 · 常见反模式深挖(Common Chart Anti-Patterns)

> 总览见 [00-roadmap.md](00-roadmap.md)。前五篇讲的是"该怎么做",这一篇收尾讲"不该怎么做"——四个
> 流传最广、最容易被审稿人一眼认出的图表反模式,每一个都不满足于"文献说这样不好",而是现场用真代码
> 把"不好在哪、坏到什么程度"量化出来:chartjunk 用真实像素计数、双 y 轴用真实"图表罪案"案例复现、
> 3D 图表用真实的透视投影复现"近大远小"、跨面板配色不一致用真实的颜色对象断言。

**本篇统一结构(七步):** 签名/是什么 → 一句话 → 底层机制(这里是感知/认知偏差的成因)→ AI 研究场景 →
可运行例子 → 审稿人/读者会怎么挑刺 → 常见坑。

---

## 1. Chartjunk——Tufte 的 data-ink ratio,用真实像素计数量化

**是什么:**
```
data-ink ratio = 承载数据信息的墨水(像素) / 图表总墨水(像素)
chartjunk: Edward Tufte 1983 年提出的术语,指"不传递任何新信息的墨水"——装饰性背景、
           过粗的边框、不必要的图例框、过密的网格线
```

**一句话:** Edward Tufte 在 1983 年《The Visual Display of Quantitative Information》里提出的
data-ink ratio(数据墨水比),把"这张图有多少装饰是纯粹的视觉噪音"变成了一个理论上可以量化的比值——
本文直接用像素计数把这个比值具体算出来,不是停留在"少即是多"这句话的层面。

**底层机制/为什么这样设计(这里是感知/认知偏差的成因):** Tufte 把"data-ink"定义为"图形里不可擦除
的核心部分,是根据数字变化而排布的、非冗余的墨水";凡是可以擦掉却不损失任何数据信息的部分,都属于
"non-data-ink",这类墨水如果多到影响读图效率,就是 chartjunk。他给出的核心设计准则是"最大化数据
墨水比""在合理范围内擦除非数据墨水/冗余数据墨水"——重点是"在合理范围内",Tufte 本人也承认图形设计
里"复杂性、结构、密度乃至美感"依然有存在空间,不是要求把图表做到极简到只剩坐标轴和数据点。**这条
原则后来也受到过实证研究的挑战**,需要诚实说明:后续的用户体验实验发现,Tufte 认为的一部分
chartjunk(比如坐标轴线)在某些场景下反而提高了读图准确率,也有研究发现适度的"chartjunk"能提升
图表的可记忆性(memorability)和读者的参与度/興趣——这些发现不是要推翻"过度装饰是坏事"这个大方向,
是提醒"data-ink ratio 和图表质量之间不是一条简单的线性关系",本文不回避这条学术界内部的后续争议。

**AI 研究场景:** 论文投稿系统/期刊排版对图表的视觉噪音格外敏感,因为版面空间有限——一张塞满装饰性
网格线、阴影图例框、渐变背景的图,不仅浪费宝贵的版面,还会在缩小到论文单栏宽度后(呼应
[03 号文件第 4 条](03-multi-panel-layout-engineering.md)的"authored size陷阱")让真正的数据线条
被这些装饰元素进一步挤压得更难辨认。

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

OKABE_ITO_BLUE = "#0072B2"

def count_ink_pixels(png_path, bg=(255, 255, 255), tol=8):
    """把 data-ink ratio 的'墨水'概念直接操作化成像素计数:和背景色差异超过阈值的像素记为'墨水'。
    这不是 Tufte 原始定义的精确实现(那需要知道哪些笔画'可以被擦掉而不丢信息',无法只从像素反推),
    是一个诚实、可计算、足以对比出差异的代理指标——差多少倍是真数出来的,不是拍脑袋估的。"""
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img).astype(int)
    diff = np.abs(arr - np.array(bg)).sum(axis=2)
    return int((diff > tol).sum()), diff.size

values = [3, 7, 5, 9]
labels = ["Q1", "Q2", "Q3", "Q4"]

# chartjunk 版本: 灰色背景填充 + 加粗黑色网格线 + 加粗边框 + 阴影图例框 -- 全部是装饰性元素,
# 编码的数据(4根柱子的高度)和clean版本完全相同
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values, color=OKABE_ITO_BLUE, edgecolor="black", linewidth=2)
ax.grid(True, which="both", axis="both", linewidth=1.2, color="black")
ax.set_facecolor("#dddddd")
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.legend(["revenue"], loc="upper left", frameon=True, shadow=True)
out_junk = os.path.join(ASSETS, "06_chartjunk.png")
fig.savefig(out_junk, dpi=150)
plt.close(fig)

# clean 版本: 同样 4 个数字,去掉装饰,只留传递数据信息必须的元素
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values, color=OKABE_ITO_BLUE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
out_clean = os.path.join(ASSETS, "06_clean.png")
fig.savefig(out_clean, dpi=150)
plt.close(fig)

junk_ink, total_px = count_ink_pixels(out_junk)
clean_ink, total_px2 = count_ink_pixels(out_clean)
assert total_px == total_px2   # 两张图尺寸/dpi相同,像素总数一致,比较才公平
assert junk_ink > clean_ink    # 编码同样4个数字,junk版本用了明显更多的"墨水"
ratio = junk_ink / clean_ink
print(f"junk version ink-pixel share: {junk_ink/total_px:.1%}")
print(f"clean version ink-pixel share: {clean_ink/total_px2:.1%}")
print(f"junk / clean ink-pixel ratio: {ratio:.2f}")
assert ratio > 1.5   # 实测约 2.0 倍,同样的 4 个数字,junk 版本几乎用了两倍的"墨水"
```

本机实测:两张图编码的是完全相同的 4 个数字,junk 版本的"墨水像素"占比约 60.8%,clean 版本约
30.1%——junk 版本用了约 **2.0 倍**的墨水表达同样的信息量,这个倍数是真实计算出来的,不是估计值。

**审稿人/读者会怎么挑刺:**
- "这张图的灰色背景和加粗网格线是想强调什么吗?我没看出这些装饰和数据有什么关系。"——审稿人一旦
  开始追问某个视觉元素"想强调什么",而回答不出实质性理由,基本等于承认那是纯装饰。
- "图例框加了阴影,这在正式发表的图表里是不是有点像 Office 默认主题的痕迹?"——阴影、渐变这类
  Office/Excel 默认图表主题常见的装饰元素,在学术图表语境下经常被认为是"没有精心处理过"的信号。
- "把这些装饰去掉,这张图看起来会不会更清楚?"——这条挑刺本身往往就是审稿意见里的具体修改建议,
  chartjunk 类反馈通常伴随一个明确可执行的修改方向,不是单纯的负面评价。

**常见坑:**
- 把 Tufte 的"最大化数据墨水比"理解成"越极简越好、必须去掉所有非数据元素"——上面"底层机制"已经
  提到后续研究对这条极简主张有实证层面的修正,坐标轴线、必要的网格线在很多场景下依然有助于读图,
  不是所有非数据墨水都该被无脑删除。
- 保留 matplotlib/Excel 的默认样式(灰色背景、密集网格线是很多绘图工具早期版本的默认主题)而没有
  意识到这些默认值本身就是相对"重"的视觉设计,不是刻意为之却依然造成了 chartjunk 的效果。
- 单一数据系列(比如本例只有一个"revenue"类别)依然画蛇添足加图例——图例本身也是一种"墨水",当
  x 轴标签已经清楚标出了每根柱子代表什么时,图例是完全冗余的信息,如上面 clean 版本所示直接省略。

---

## 2. 双 y 轴陷阱——真实"图表罪案"案例与修复手法

**是什么:**
```
ax2 = ax1.twinx()   # 在同一个 x 轴上,叠加第二个独立缩放的 y 轴
```

**一句话:** 双 y 轴图允许两条曲线各自独立缩放,这个"独立"正是问题所在——制图者(有意或无意)可以
分别调整两个轴的范围,制造出两条数据毫不相关、却"看起来完美同步"的视觉假象,数据可视化圈子里流传的
几个真实"图表罪案"案例,几乎都是靠这个机制制造出误导效果的。

**底层机制/为什么这样设计(这里是认知偏差的成因):** 人眼阅读折线图时,天然会去关注"两条线的形状是否
同步变化",却很少会先去确认"这两个 y 轴的刻度范围是不是特意设定过"——这是一种视觉上的捷径:形状匹配
比"先检查两个坐标系是否公平"更容易、更快被大脑处理,双 y 轴图正是利用了这个认知捷径。真实案例:圣路易斯
联邦储备银行曾发布一张军费开支双轴图,被社交媒体大量质疑为"图表罪案"(chart crime),因为独立缩放
的两条轴制造出"中国军费已经反超美国"的错误印象;另一起案例中,Americans United for Life 发布的
双轴图被数据可视化社区批评为"不可原谅的误导",因为图上固定了一个不从零开始的 Y 轴范围、又没有标出
具体刻度数字,进一步放大了视觉上的变化幅度。修复手法通常有两种:要么把两条线拆成两张独立的子图分别
展示;要么把两个系列都换算成"相对某个基准年份的百分比变化"(归一化到同一个指数刻度),让它们能公平
共享同一个 y 轴。

**AI 研究场景:** 训练过程中同时监控"loss"和"某个业务指标(比如任务成功率)"随训练步数变化,是最
容易诱发双 y 轴滥用的场景——loss 和成功率的数值范围天差地别(loss 可能是 0-5,成功率是 0-1),直觉
上会想用双轴各自缩放来"都放进同一张图里",但如果不谨慎处理刻度范围,很容易无意中制造出"这两者高度
相关"的误导性视觉印象,即使两者可能只是恰好同时在训练过程中数值上下波动。

**可运行例子:**
```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00"}

years = np.array([1, 2, 3, 4, 5, 6])
metric_a = np.array([10, 12, 11, 13, 12, 14])     # 真实波动幅度很小(10-14区间)
metric_b = np.array([100, 98, 102, 95, 101, 90])   # 真实波动幅度也很小(90-102区间)

# 陷阱版本: 双 y 轴各自独立缩放到很窄的范围,把两者本来很小的波动都视觉放大成大起大落
fig, ax1 = plt.subplots(figsize=(4, 3))
ax2 = ax1.twinx()
ax1.plot(years, metric_a, color=OKABE_ITO["blue"], marker="o")
ax2.plot(years, metric_b, color=OKABE_ITO["vermillion"], marker="s")
ax1.set_ylim(9, 15)     # metric_a 真实范围是 10-14,这里几乎贴着数据范围画,视觉波动被放大
ax2.set_ylim(85, 105)   # metric_b 同理
out_bad = os.path.join(ASSETS, "06_dualaxis_bad.png")
fig.savefig(out_bad, dpi=150)
plt.close(fig)

# 修复版本: 都换算成"相对第1年的百分比"(索引化),公平共享同一个 y 轴,不再需要双轴
idx_a = 100 * metric_a / metric_a[0]
idx_b = 100 * metric_b / metric_b[0]
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(years, idx_a, color=OKABE_ITO["blue"], marker="o", label="metric A (index)")
ax.plot(years, idx_b, color=OKABE_ITO["vermillion"], marker="s", label="metric B (index)")
ax.axhline(100, color="black", linewidth=0.8, linestyle=":")
ax.set_ylabel("% of baseline (year 1 = 100)")
ax.legend(frameon=False, fontsize=8)
out_fixed = os.path.join(ASSETS, "06_dualaxis_fixed.png")
fig.savefig(out_fixed, dpi=150)
plt.close(fig)

assert os.path.exists(out_bad) and os.path.getsize(out_bad) > 0
assert os.path.exists(out_fixed) and os.path.getsize(out_fixed) > 0

# 用真实数字说明"陷阱版本"的视觉放大有多严重: metric_a 真实变化幅度只有 40%(10到14),
# 但如果 y 轴范围贴着数据画(9-15,总跨度只有6),视觉上这40%的变化几乎占满了整个纵轴高度
true_range_pct = (metric_a.max() - metric_a.min()) / metric_a.min()
visual_axis_span = 15 - 9
print(f"metric_a true relative range: {true_range_pct:.1%}, trap-version y-axis span is only {visual_axis_span} units")
assert true_range_pct < 0.5   # 真实波动不到50%
```

**审稿人/读者会怎么挑刺:**
- "这两个 y 轴的范围是怎么定的?为什么不干脆从 0 开始?"——一旦审稿人开始追问坐标轴范围的选择依据,
  制图者往往很难给出"这不是为了制造某种视觉效果"的令人信服的解释。
- "这两条曲线看起来高度同步,但它们本来就应该相关吗,还是只是坐标轴凑出来的巧合?"——双轴图最容易
  引发的质疑正是"这个视觉相关性有没有统计学意义上的支持",而不只是两条线形状恰好相似。
- "能不能拆成两张图分别看?我想知道每个指标各自真实的变化幅度有多大。"——这是最常见、也最容易被
  采纳的修改建议,直接对应上面演示的两种修复手法之一。

**常见坑:**
- 双 y 轴本身不是绝对不能用的禁忌(某些场景下,比如物理量纲完全不同但确实有内在关联的两个变量,双轴
  依然是合理选择),但必须确保两个轴的刻度范围有一个说得清楚、经得起追问的选择依据,不能只是"调到
  两条线看起来重合"。
- 忘记在坐标轴标签/图注里清楚标出"左轴对应哪条线、右轴对应哪条线"——即使双轴的使用本身站得住脚,
  读者也需要显式的视觉提示(通常是把轴标签颜色和对应曲线颜色对应起来)才能正确解读。
- 把"索引化/归一化到百分比"这个修复手法用在不适合归一化的场景——如果两个指标的绝对数值本身就是
  论点的一部分(比如要展示"提升了整整 10 个百分点"而不是"相对提升了多少倍"),归一化后反而丢失了
  读者需要的绝对数值信息,拆成两张独立子图有时候是更合适的修复手法。

---

## 3. 3D 图表的透视失真——用 `bar3d` 复现"近大远小"

**是什么:**
```
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type('persp', focal_length=...)   # 短焦距 = 更强的透视畸变(类似广角镜头)
ax.bar3d(x, y, z, dx, dy, dz, color=...)
```

**一句话:** 3D 饼图/3D 柱状图最根本的问题不是"看起来花哨",是**透视投影本身会系统性扭曲物体的视觉
大小**——离视角更近的元素被放大、更远的元素被压缩/遮挡,这意味着哪怕两个数据切片/柱子的真实数值
完全相等,3D 视角下也会呈现出明显不同的视觉大小,下面直接用代码构造四个数值完全相等的柱子,让这个
失真效果在真实渲染里现出原形。

**底层机制/为什么这样设计(这里是感知/认知偏差的成因):** 人眼理解 3D 场景时会自动做"近大远小"的
深度线索补偿——这是日常生活里判断物体真实大小的必要机制(远处的车看起来小,但我们知道它实际大小
没变),但这套补偿机制被直接套用到"用大小表示数值"的图表上时,恰恰会产生系统性误判:图表设计者的
本意是"用视觉大小精确编码数值",而 3D 透视会让同样大小的两个数据元素因为摆放位置的远近不同,呈现出
不同的视觉大小,这和"用位置远近表示深度"的日常经验冲突——最终结果是被前景元素占据主导,后方元素
被视觉上低估。真实的实证发现:同一份数据的 2D 与 3D 饼图对比里,3D 版本会让两个真实大小相同的切片
看起来明显不同,原因是它们在圆盘上所处的角度让其中一个更朝向观众、看起来占据更大面积。

**AI 研究场景:** 这条反模式几乎不会以"3D 饼图"的原始形态出现在严肃的 AI 论文里(经验丰富的审稿人
一眼就会打回),但它的变体值得警惕——比如"伪 3D"柱状图(matplotlib/Excel 里给普通 2D 柱状图加一层
立体阴影效果,让柱子看起来有厚度)、或者用真 3D 坐标系展示本可以用 2D 热力图/等高线图更清楚表达的
三变量数据,这些场景下"3D 让图看起来更高级"的直觉,和这里揭示的失真风险是同一个问题的不同外衣。

**可运行例子:**
```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3d 投影,即使不显式使用这个名字也需要导入)

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00", "bluish_green": "#009E73", "orange": "#E69F00"}

# 4 根柱子,数值(dz)完全相同,只是沿深度方向(y)摆放在不同距离
fig = plt.figure(figsize=(4, 3.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type('persp', focal_length=0.2)   # 短焦距,类似广角镜头,放大透视畸变效果以便观察
xpos = np.array([0, 0, 0, 0])
ypos = np.array([0, 2, 4, 6])   # 距观察者的远近依次递增
zpos = np.zeros(4)
dx = np.full(4, 1.2)
dy = np.full(4, 1.2)
dz = np.full(4, 5.0)            # 全部四根柱子,数值完全相等
colors_list = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"], OKABE_ITO["orange"]]
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_list, shade=True)
ax.view_init(elev=15, azim=-80)
ax.set_box_aspect((1, 3, 1.2))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("all four bars encode the exact same value (5.0)", fontsize=9)
out = os.path.join(ASSETS, "06_bar3d_perspective.png")
fig.savefig(out, dpi=150)
plt.close(fig)

assert os.path.exists(out) and os.path.getsize(out) > 0
assert (dz == 5.0).all()   # 四个数值断言完全相等——图里呈现出的大小差异纯粹是透视投影的产物
print("bar3d perspective-distortion demo saved, all four bar values are exactly equal:", dz.tolist())

# 一个额外的、撰写时现场发现的真实细节: matplotlib 没有为"3D 饼图"这个反模式提供任何一等公民支持
fig2 = plt.figure()
ax3d = fig2.add_subplot(111, projection="3d")
try:
    ax3d.pie([30, 20, 50])
    pie_on_3d_actually_works = True
except TypeError:
    pie_on_3d_actually_works = False
plt.close(fig2)
assert pie_on_3d_actually_works is False
print("Axes3D.pie() 'looks' callable via inheritance, but actually raises TypeError -- not a real 3D pie chart implementation")
```

**这条"常见坑"是撰写时现场发现、不是提前设计好的**:`Axes.pie()` 因为类继承关系,在 `Axes3D` 对象
上"看起来"可以调用(`hasattr(Axes3D, "pie")` 返回 `True`),但真的调用会直接抛出 `TypeError`——
`pie()` 内部会调用 `.text()` 绘制百分比标注,而 `Axes3D` 覆盖了一个签名完全不同的三维版 `.text()`,
两者不兼容导致崩溃。这其实是一个很有意思的旁证:一个被数据可视化社区公认为反模式的图表类型,连
matplotlib 自己都没有真正实现过,不是因为"技术上做不到"(用 `Poly3DCollection` 手工拼出挤出体确实
能画出一个真 3D 饼图),而是没有人认真为它写过一等公民支持——这本身就是"这个图表类型不值得投入
工程精力"的一种间接信号。

**审稿人/读者会怎么挑刺:**
- "这几个 3D 柱子/切片的真实数值差多少?光看图我判断不出来。"——3D 透视失真最直接的后果就是读者
  没法信任自己从图上直接读出的相对大小关系,必须去查数值标注或者原始表格才能确认,这本身就否定了
  图表"一眼看出大小关系"的核心价值。
- "为什么要用 3D 视角?这几个数据没有第三个维度需要表达。"——如果数据本身只有两个维度(类别 + 数值),
  引入第三个视觉维度(深度)没有对应任何真实数据,纯粹是装饰性的,是[本文第 1 条]
  (#1-chartjunk--tufte-的-data-ink-ratio用真实像素计数量化)chartjunk 概念的一个立体版本。

**常见坑:**
- 使用绘图软件"美化"选项给 2D 图表自动加上 3D 立体效果(常见于 Excel/PowerPoint 默认图表模板),
  没有意识到这个"美化"选项本身就是在系统性引入视觉失真,不是纯粹的审美升级。
- 即使意识到 3D 饼图不该用,却依然用"3D 柱状图"处理只有类别+单一数值的数据——3D 视角只有在真的
  存在第三个数据维度需要表达时才有意义(比如本条例子里用深度表示"第几次实验"这种额外类别),否则
  和 3D 饼图是同一类问题的不同外形。
- 高估读者(包括审稿人自己)在快速浏览时识别透视失真的能力——上面的实测演示证明,即使是数值完全
  相等的四个柱子,肉眼在 3D 视角下也很难认为它们"看起来一样大",这不是"读者不够细心",是这类视觉
  编码本身就在系统性地对抗准确的数值判断。

---

## 4. 跨 panel 配色不一致——多面板图必须共享同一套语义配色

**是什么:**
```
COLOR_BY_METHOD = {"baseline": "#0072B2", "ours": "#D55E00"}   # 一份贯穿全图的颜色字典
# 每个面板取色时都从这一份字典里查,不在每个面板里各自决定"这次用什么颜色画baseline"
```

**一句话:** 同一篇论文/同一张多面板图里,"baseline"这个类别如果在面板 A 是蓝色、在面板 B 变成了
红色,读者的第一反应不是"这两个面板讲的是两件独立的事",而是"这是不是编辑失误"——多面板图的配色
必须由一份共享的"类别→颜色"映射统一驱动,不能每个面板独立决定用什么颜色。

**底层机制/为什么这样设计(这里是感知/认知偏差的成因):** 颜色在图表里承担的是"符号"功能——一旦
读者在面板 A 里学会"蓝色代表 baseline",这个映射关系会被直接带到面板 B 的阅读过程中,这是格式塔
心理学里"一致性"原则的直接应用(相同的视觉符号应该始终代表相同的语义)。如果面板 B 里蓝色变成了
代表"ours",读者要么没有意识到变化、直接用面板 A 学到的错误映射去解读面板 B(得出完全相反的结论),
要么发现了不一致、但因此需要重新逐个面板核对图例,阅读成本大幅上升——无论哪种情况,这个反模式造成
的后果都比"配色本身不好看"严重得多,是一个正确性风险,不只是审美问题。

**AI 研究场景:** [07 号教程体文件](07-build-a-mini-publication-figure.md)最终画的核心结果图,双
面板(性能曲线 + 汇总对比柱状图)共用同一批方法(baseline/ours),这正是本条反模式最容易发生、也
最需要提前规避的场景——一份显式的颜色字典(而不是在每个面板各自调用 `plt.bar(..., color=...)` 时
凭记忆敲十六进制值)是最简单可靠的规避手段。

**可运行例子:**
```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00"}

# 反模式版本: 两个面板各自决定配色顺序,"ours"在面板A是vermillion,在面板B变成了blue
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6, 3))
axL.bar(["baseline", "ours"], [0.5, 0.7], color=[OKABE_ITO["blue"], OKABE_ITO["vermillion"]])
axL.set_title("panel A (ours = vermillion)", fontsize=9)
axR.bar(["baseline", "ours"], [0.4, 0.6], color=[OKABE_ITO["vermillion"], OKABE_ITO["blue"]])
axR.set_title("panel B (ours = blue)", fontsize=9)
out_bad = os.path.join(ASSETS, "06_cross_panel_bad.png")
fig.savefig(out_bad, dpi=150)
plt.close(fig)
assert os.path.exists(out_bad) and os.path.getsize(out_bad) > 0

# 修复版本: 一份共享的"类别->颜色"字典驱动全部面板取色
COLOR_BY_METHOD = {"baseline": OKABE_ITO["blue"], "ours": OKABE_ITO["vermillion"]}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6, 3))
methods = ["baseline", "ours"]
barsL = axL.bar(methods, [0.5, 0.7], color=[COLOR_BY_METHOD[m] for m in methods])
axL.set_title("panel A", fontsize=9)
barsR = axR.bar(methods, [0.4, 0.6], color=[COLOR_BY_METHOD[m] for m in methods])
axR.set_title("panel B", fontsize=9)
out_fixed = os.path.join(ASSETS, "06_cross_panel_fixed.png")
fig.savefig(out_fixed, dpi=150)
plt.close(fig)
assert os.path.exists(out_fixed) and os.path.getsize(out_fixed) > 0

# 这一条不止能验证"文件生成了",还能直接断言"一致性"这件事本身:
# 从两个面板各自的 bar patch 对象里读出真实渲染用的 facecolor,确认"ours"在两个面板里是同一个颜色
ours_color_panel_a = barsL[1].get_facecolor()
ours_color_panel_b = barsR[1].get_facecolor()
assert ours_color_panel_a == ours_color_panel_b == mcolors.to_rgba(OKABE_ITO["vermillion"])
print("cross-panel consistency assertion passed, 'ours' rendered color (RGBA) in both panels:", ours_color_panel_a)
```

**审稿人/读者会怎么挑刺:**
- "面板 A 里蓝色代表 baseline,面板 B 里蓝色怎么变成你们的方法了?这两个面板是不是配色对调了?"——
  一旦被审稿人问到这种问题,几乎无法用"审美选择"来辩解,只能承认是疏漏。
- "整篇论文里,'我们的方法'在图 2 是红色,在图 5 又变成了绿色,能不能统一一下?"——这条反模式不
  只发生在同一张多面板图内部,同一篇论文里不同图之间的配色不一致是更容易被忽视、但同样会造成困扰
  的扩展版本。
- "你们是用什么流程生成这些图的?每张图是不是独立脚本、独立调用配色参数?"——审稿人这类追问背后
  真正关心的是"这篇论文的图表生产流程是否严谨可控",配色不一致往往是这类流程缺乏统一规范的外在
  信号。

**常见坑:**
- 在生成多张图/多个面板的代码里,每次画图都重新手写一遍十六进制颜色值或者依赖 matplotlib 的默认
  颜色循环(`color cycle`)——默认颜色循环是按"这是本次绘图调用里第几条线/第几根柱子"分配颜色,
  和"这个类别语义上应该是什么颜色"完全无关,面板之间调用顺序稍有不同,分配到的颜色就会跟着错位。
- 团队协作时,不同成员各自负责论文里的不同图,没有共享一份统一的配色字典/配置文件,导致整篇论文
  完成后才发现配色风格各自为政。
- 只检查了"每张图自己看起来配色搭配得当",没有把全部图表并排放在一起做一次"跨图一致性"的复查——
  这一步很容易在赶稿阶段被跳过,但恰恰是审稿人会做的事(把所有图放在一起对比着看)。

---

*创建:2026-07-25*
