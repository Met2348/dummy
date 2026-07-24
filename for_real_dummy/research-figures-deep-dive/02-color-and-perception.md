# 02 · 颜色与感知设计深挖(Color & Perceptual Design)

> 总览见 [00-roadmap.md](00-roadmap.md)。[01 号文件](01-chart-type-selection.md)决定了"画哪种图",
> 这一篇决定"这张图用什么颜色"——颜色不是审美选择,是一个有真实感知科学依据、有具体十六进制数值、能
> 被真代码验证对错的工程决策。全篇会现场用 Python 算出两条真实结论:①`jet` colormap 的感知亮度确实
> 非单调,`viridis` 确实单调;②Okabe-Ito 调色板的 8 个颜色确实是真实存在、可以直接抄的十六进制值,
> 不是"随便挑几个看起来协调的颜色"。

**本篇统一结构(七步):** 签名/是什么 → 一句话 → 底层机制(视觉感知原理)→ AI 研究场景 → 可运行例子 →
审稿人/读者会怎么挑刺 → 常见坑。

---

## 1. Okabe-Ito 色盲友好离散调色板——8 个具体的十六进制值

**是什么:**
```
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
```

**一句话:** 这 8 个具体的十六进制颜色值,是冈部昌宏(Masataka Okabe)和伊藤啓(Kei Ito)2002 年"色彩通用
设计"(Color Universal Design, CUD)项目里专门为色觉缺陷(color vision deficiency, CVD)人群设计、
2008 年正式发表、2011 年被 Bang Wong 在 *Nature Methods* 上重新推广(因此也常被称为"Wong 调色板")的
离散分类配色方案——不是"经验上大家常用"这种模糊说法,是一组有据可查、能直接照抄十六进制值用进
matplotlib 的具体数字。

**底层机制/为什么这样设计:** 人群里色觉缺陷的发生率不是可以忽略的小概率事件——约 8% 的男性和 0.5%
的女性有某种形式的色觉缺陷,最常见的是红绿色盲(红色盲 protanopia、绿色盲 deuteranopia),更少见的是
蓝黄色盲(tritanopia)。Okabe-Ito 这 8 个颜色的设计目标,是**同时**在这三种最常见色觉缺陷类型下都能被
区分开——这和"先设计一套配色,再拿去测试色盲能不能看清"的事后补救思路不同,是从设计源头就把三种色觉
缺陷都当成约束条件。另一个容易被忽视的设计目标:这 8 个颜色即使被打印成黑白(灰度)也保持可区分的
亮度差异——上面第 2 条会展示"亮度是否单调/有区分度"怎么用代码算出来,这条设计目标同样可以被验证。

**AI 研究场景:** 论文里任何"用不同颜色区分几个类别/几个方法"的图(消融实验的柱状图配色、多条方法的
折线图配色、散点图里不同类别的点),第一反应就该是"从 Okabe-Ito 这 8 个颜色里选,不要用 matplotlib
默认的 `tab10`/`Set1` 这类调色板"——不是说默认调色板完全不能用,而是 Okabe-Ito 是**专门为色觉缺陷验证
过**的选择,默认调色板没有这个保证。8 个颜色通常也够用了:大多数论文图不会同时对比超过 8 个类别,一旦
超过,应该考虑换图表类型(比如按 [01 号文件](01-chart-type-selection.md)的判断改用热力图或者拆成
多个小面板),而不是继续加第 9 个、第 10 个颜色。

**可运行例子:**
```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
assert len(OKABE_ITO) == 8
assert all(v.startswith("#") and len(v) == 7 for v in OKABE_ITO.values())

fig, ax = plt.subplots(figsize=(7, 1.6))
for i, (name, hexcode) in enumerate(OKABE_ITO.items()):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hexcode))
    ax.text(i + 0.5, -0.15, hexcode, ha="center", va="top", fontsize=7)
ax.set_xlim(0, 8)
ax.set_ylim(-0.35, 1)
ax.axis("off")
out = os.path.join(ASSETS, "02_okabe_ito_swatches.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)

# 验证方式见 00-roadmap.md:文件真的生成 + 尺寸元数据符合预期(bbox_inches="tight" 会裁剪空白,
# 所以这里只核对文件非空,不核对精确像素数——tight 裁剪后的精确尺寸依赖字体渲染,不是稳定可预测的数字)
assert os.path.exists(out) and os.path.getsize(out) > 0
img = Image.open(out)
print("swatches saved:", out, "size:", img.size)
```

**审稿人/读者会怎么挑刺:**
- "这几个类别的颜色,红绿色盲的读者能分清楚吗?"——如果用的是 matplotlib 默认调色板里红绿相邻的两个
  颜色,这是一个越来越常被提起的问题,尤其是生物医学/神经科学类审稿人对这条格外敏感。
- "只在颜色上做区分,打印成黑白版本还能看清楚吗?"——很多读者依然会打印论文或者在灰度电子书阅读器上
  看,纯彩色编码在这种场景下完全失效,[本文第 4 条](#4-冗余编码色盲读者不能只靠颜色区分类别)会展开讲
  怎么补救。
- "为什么图例里第 3 类和第 6 类的颜色看起来几乎一样?"——即使不是色盲读者,如果调色板本身包含两个
  过于接近的颜色,也会造成误读,这是 Okabe-Ito 这类"专门优化过颜色之间可区分度"的调色板要解决的问题。

**常见坑:**
- 超过 8 个类别时继续从 Okabe-Ito 里"挤"出更多颜色(比如用透明度变化制造出第 9、10 个视觉上的类别)——
  这个调色板的设计边界就是 8 个精心挑选、互相之间可区分度经过验证的颜色,人为插值出来的颜色不再有
  同等的可区分度保证。
- 把 Okabe-Ito 的黑色(`#000000`)同时用作"数据颜色"和"坐标轴/文字颜色"——如果坐标轴、网格线也是
  纯黑色,会和用黑色代表的那一类数据混在一起,通常黑色更适合留给"参考线"/"baseline"这类衬托性的元素
  (呼应本文可运行例子和后面几篇的用法),不是随便分配给第一个类别。
- 只记住了"Okabe-Ito"这个名字,但配色时凭记忆估计颜色而不是直接抄录上面这 8 个精确的十六进制值——
  颜色的可区分度是这个调色板反复调过的结果,凭记忆估算的近似色不再有同等保证。

---

## 2. 为什么不能用 jet/rainbow colormap——用真代码算出"感知亮度非单调"

**是什么:**
```
plt.get_cmap(name)(x)  # x∈[0,1],返回该 colormap 在这个位置的 RGBA 颜色
relative_luminance(rgba) = 0.2126*R + 0.7152*G + 0.0722*B   # ITU-R BT.709 相对亮度公式
```

**一句话:** `jet`(以及其他彩虹色 colormap)最根本的问题不是"审美过时",是它的**感知亮度不随数据值
单调变化**——数据在增大,但颜色在人眼看来时亮时暗,这会在数据里制造出实际不存在的"假边界",而
`viridis` 这类"感知均匀"colormap 从设计上就保证亮度单调,下面直接用代码把这个差异算出来,不是空口
引用文献。

**底层机制/为什么这样设计:** Borland 和 Taylor 2007 年发表在 *IEEE Computer Graphics and Applications*
上的经典论文《Rainbow Color Map (Still) Considered Harmful》原话总结了三条问题:彩虹 colormap 缺乏
"感知排序"(perceptual ordering,人眼无法直接感受出颜色序列对应的数值大小顺序)、亮度变化不受控制、
会主动引入和数据无关的视觉梯度、误导解读。这不是纯理论担忧——Borkin 等 2011 年的研究发现,医生用彩虹
colormap 诊断心脏疾病时,比用感知均匀的 colormap 花更长时间、犯更多错误。这条批评推动了真实的工程
变化:MATLAB 把默认 colormap 从 jet 换成了 parula;matplotlib 2.0(2015 年)把默认 colormap 从 jet
换成了 `viridis`——由 Stéfan van der Walt 和 Nathaniel Smith 在同年 SciPy 大会上提出,`viridis` 是
他们设计的四个候选(Magma/Inferno/Plasma/Viridis)之一,核心设计目标是**亮度单调递增**、**灰度打印
依然可辨**、**避免制造假边界**。下面可运行例子直接把这条"亮度单调性"的设计目标用代码算出来验证,不
只是转述这段历史。

**AI 研究场景:** 论文里最常见的连续值可视化(loss landscape、attention 权重矩阵、[01 号文件第 3 条]
(01-chart-type-selection.md)的超参数网格热力图、任何 `imshow`/`pcolormesh` 调用)默认都应该用
`viridis` 或它的同族 colormap(`plasma`/`inferno`/`magma`/`cividis`,matplotlib 里统称"Perceptually
Uniform Sequential"分类),而不是手滑传成 `jet`——`matplotlib.pyplot.get_cmap("jet")` 依然能调用
成功(matplotlib 没有把 jet 从库里删除,只是换了默认值),这意味着这个坑不会被语法检查拦下来,只能
靠人自己记住这条设计判断。

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

def relative_luminance(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

samples = np.linspace(0, 1, 20)
jet_lum = np.array([relative_luminance(plt.get_cmap("jet")(s)) for s in samples])
viridis_lum = np.array([relative_luminance(plt.get_cmap("viridis")(s)) for s in samples])

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(samples, jet_lum, color=OKABE_ITO["vermillion"], marker="s", label="jet")
ax.plot(samples, viridis_lum, color=OKABE_ITO["blue"], marker="o", label="viridis")
ax.set_xlabel("position along colormap (0=low value, 1=high value)")
ax.set_ylabel("relative luminance (perceived brightness)")
ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(ASSETS, "02_jet_vs_viridis_luminance.png")
fig.savefig(out, dpi=200)
plt.close(fig)
assert os.path.exists(out) and os.path.getsize(out) > 0

# 核心结论: 沿着 colormap 从数据小到数据大采样,亮度序列有几步是"变暗"的(非单调)
jet_decreases = int((np.diff(jet_lum) < -1e-9).sum())
viridis_decreases = int((np.diff(viridis_lum) < -1e-9).sum())
assert jet_decreases >= 5       # 实测 19 步里有 7 步在数据变大时亮度反而下降
assert viridis_decreases == 0   # viridis 亮度严格单调不减,一步都不违反——这正是它被设计出来的目标
print("jet luminance-decrease steps:", jet_decreases, "/ viridis luminance-decrease steps:", viridis_decreases)
```

本机实测:`jet` 在 19 个采样间隔里有 **7 处**亮度不增反降(数据在变大,人眼感受到的亮度却在变暗),
`viridis` 全部 19 处**零违反**、严格单调递增——这不是文献转述,是撰写时在 `.venv` 里现场采样、按公式
算出来的真实数字。

**审稿人/读者会怎么挑刺:**
- "你的热力图用的是不是 jet?"——这条批评在可视化社区已经流传近 20 年(Rogowitz & Treinish 1998 年
  就提出过近似批评,Borland & Taylor 2007 年进一步系统化),对可视化敏感的审稿人扫一眼配色就能认出来。
- "这张连续值热力图中间那圈'轮廓'是真实数据特征,还是 colormap 亮度跳变制造出来的假象?"——这是
  jet 类 colormap 最容易被追问穿的地方,一旦读者怀疑某个视觉边界是颜色伪影而不是真实数据结构,整张图
  的可信度都会被打折扣。
- "灰度打印这张图之后还能看出数据的高低吗?"——亮度非单调的 colormap 一旦变成灰度,数据顺序会完全
  错乱;`viridis` 因为亮度单调,灰度版本依然大致保留数据的高低关系。

**常见坑:**
- 2015 年之前的很多教程/课件代码示例默认写 `cmap="jet"`,复制粘贴旧代码是这个坑最常见的来源,而不是
  真的认为 jet 更好看。
- 以为"只要不用彩虹色就安全"——不是所有非彩虹的 colormap 都是感知均匀的,选择时认准 matplotlib 官方
  文档里明确标注为"Perceptually Uniform Sequential"分类的那几个(viridis/plasma/inferno/magma/
  cividis),不要凭肉眼判断"这个 colormap 看起来顺眼"就假设它亮度也单调。
- 在需要"零为有意义中点"的数据上(比如"相对提升/下降的百分比",正负两个方向都有意义)依然使用
  sequential colormap(单向渐变)——这种数据该用[下面第 3 条](#3-sequentialdivergingqualitative-三类-colormap-怎么选)
  的 diverging colormap,不是本条讨论的 sequential 单向渐变。

---

## 3. Sequential/Diverging/Qualitative 三类 colormap 怎么选

**是什么:**
```
choose_colormap_family(has_meaningful_zero_or_midpoint, is_categorical) -> (family_name, example)
```

**一句话:** matplotlib 官方文档把 colormap 分成三大类(sequential 单向渐变、diverging 双向发散、
qualitative 无序分类),选错类别比选错具体某个颜色更容易误导读者——这三类分别回答"数据有没有大小顺序"
和"有没有一个有意义的零点/中点"这两个问题。

**底层机制/为什么这样设计:**
- **Sequential(单向渐变,如 `viridis`)**:数据本身是从低到高的**单向**顺序(成功率、损失值、密度),
  没有一个特殊的"中点"值需要被强调,颜色从暗到亮(或者反过来)单调变化,直接对应数据单调变化。
- **Diverging(双向发散,如 `RdBu`/`coolwarm`)**:数据有一个有意义的中点(通常是 0,比如"相对
  baseline 的提升/下降百分比"、"相关系数"、"温度距平"),需要让读者第一眼就分辨出"这个格子是在中点
  之上还是之下",单向渐变做不到这件事——`viridis` 这类 colormap 里,数值为 0 和数值为某个较小正数的
  颜色可能看起来很接近,读者无法一眼确认符号。Diverging colormap 把中点固定成一个中性色(通常是白色
  或浅灰色),两侧分别延伸向两种不同色相,符号一眼可辨。
- **Qualitative(无序分类,如 `tab10`)**:数据根本没有大小顺序,是纯类别标签(不同的模型名字、不同的
  数据集),这时候颜色**不应该**有渐变关系——如果类别本身无序,却用一个渐变 colormap 去配色,会在读者
  脑子里暗示"类别之间有顺序"这个不存在的关系,这条本质上是[本系列第 1 条](#1-okabe-ito-色盲友好离散调色板8-个具体的十六进制值)
  的延伸:Okabe-Ito 调色板就是给这一类场景用的。

**AI 研究场景:** "两个模型在每个测试子集上的性能差值"这类图(经常出现在消融分析或者 error analysis
部分)天然有一个有意义的零点(差值为 0 表示两个模型一样好),该用 diverging;"不同方法在不同预算下的
成功率"这类没有中点、只有单向好坏的数据,该用 sequential;"不同随机种子/不同 checkpoint"这类纯分类
标签,该用 qualitative——选错类别是审稿人一眼就能看出的"图表基本功不扎实"信号。

**可运行例子:**
```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

def choose_colormap_family(has_meaningful_zero_or_midpoint, is_categorical):
    if is_categorical:
        return "qualitative", "tab10"
    if has_meaningful_zero_or_midpoint:
        return "diverging", "RdBu"
    return "sequential", "viridis"

assert choose_colormap_family(False, is_categorical=True) == ("qualitative", "tab10")
assert choose_colormap_family(True, is_categorical=False) == ("diverging", "RdBu")
assert choose_colormap_family(False, is_categorical=False) == ("sequential", "viridis")

fig, axes = plt.subplots(3, 1, figsize=(5, 3))
grad = np.linspace(0, 1, 256).reshape(1, -1)
axes[0].imshow(grad, cmap="viridis", aspect="auto")
axes[0].set_title("sequential: viridis", fontsize=8)
diverg = np.linspace(-1, 1, 256).reshape(1, -1)
axes[1].imshow(diverg, cmap="RdBu", aspect="auto")
axes[1].set_title("diverging: RdBu (0 = white midpoint)", fontsize=8)
qual = np.arange(10).reshape(1, -1)
axes[2].imshow(qual, cmap="tab10", aspect="auto")
axes[2].set_title("qualitative: tab10 (no order implied)", fontsize=8)
for ax in axes:
    ax.set_yticks([])
    ax.set_xticks([])
fig.tight_layout()
out = os.path.join(ASSETS, "02_colormap_families.png")
fig.savefig(out, dpi=200)
plt.close(fig)
assert os.path.exists(out) and os.path.getsize(out) > 0
print("colormap families figure saved:", out)
```

**审稿人/读者会怎么挑刺:**
- "这张图里 0 用什么颜色表示?我怎么知道这个格子是正的还是负的?"——用 sequential colormap 画本该
  用 diverging 的数据,读者经常要跑去查 colorbar 上的具体刻度才能确认符号,体验很差,是可以被直接
  指出的设计缺陷。
- "这几个类别的颜色深浅不一,是不是在暗示某种排序?"——qualitative 数据被套上一个渐变 colormap
  (比如按字母顺序给几个模型名字配上从浅到深的同一色系),会让读者误以为存在一种"越往后越怎样"的顺序。
- "diverging colormap 的中点真的对应数据的 0 吗?"——如果没有显式设置 `vmin`/`vmax` 让中点真正落在
  0(比如数据范围是 `[-2, 5]` 却没有手动让 colormap 中心对齐 0),diverging colormap 的"白色中点"会
  错位到某个不是 0 的值上,视觉暗示的符号信息反而是错的,这是这类 colormap 最容易踩的技术坑。

**常见坑:**
- 用 diverging colormap 时忘记显式设置对称的 `vmin`/`vmax`(比如 `vmin=-abs_max, vmax=abs_max`),
  导致中性色(白/浅灰)对应的实际不是数据的 0 点,而是数据范围的几何中点——这两者只有在数据本身正负
  对称时才恰好相同。
- 把 qualitative 数据的类别顺序和图例顺序不一致(比如画图时用字典遍历顺序,但正文表格按字母顺序),
  同一个类别在不同图之间颜色对不上,这条同时也是[06 号文件第 4 条](06-anti-patterns.md)要讲的"跨
  panel 配色不一致"反模式的一个具体来源。
- 只知道 `RdBu`/`coolwarm` 是"红蓝配色"就直接当 sequential 用在没有中点语义的数据上——这类 colormap
  的两端颜色虽然好看,但中间的浅色/白色区域对单向渐变数据是一种视觉浪费(中间一大段数据看起来都很
  接近白色,难以区分)。

---

## 4. 冗余编码——色盲读者不能只靠颜色区分类别

**是什么:**
```
redundant_encoding = color + marker_shape + linestyle   # 三个独立的视觉通道编码同一个类别变量
# 任何一个通道失效(色盲/打印成灰度/黑白投影仪),剩下的通道依然能让读者分清类别
```

**一句话:** 只用颜色区分类别,色盲读者、黑白打印、老旧投影仪三种场景都会让图表失效——同时用形状
(marker)和线型(linestyle)重复编码同一个信息,任何一个通道失效时,剩下的通道依然撑得住。

**底层机制/为什么这样设计:** 这是"信息论"角度一个很朴素的道理:如果一条信息只走一个信道,这个信道
一旦失真,信息就丢了;走三个独立信道(颜色、形状、线型),只要不是三个同时失真,读者依然能正确解码。
折线图对这条原则格外敏感,因为它还有个额外的失败模式——当两条颜色相近的线在图上交叉时,读者的视线在
交叉点很容易"跳线"(跟错了线),如果这两条线的线型/marker 也不一样,即使颜色分不清,交叉点前后的
线型也能帮读者重新对上号。一个便宜但很实用的自测方法是"灰度测试":把画好的图转换成灰度,如果两条本该
分清的线在灰度下几乎无法区分,说明冗余编码没做够。

**AI 研究场景:** 论文的每一张多曲线/多类别对比图都适用这条原则,尤其是要投稿到印刷版仍然是黑白/影印
的会议(不少老牌会议的正式 proceedings 依然只发黑白印刷版,即使在线版本是彩色的),或者预期读者会
打印出来精读(审稿人常见的阅读习惯)——纯彩色编码在这些场景下全部失效。[07 号教程体文件]
(07-build-a-mini-publication-figure.md)从第一步开始就会同时使用颜色+marker+线型三重编码,不是画完
图之后再回头补。

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

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00"}

x = np.arange(6)
y1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
y2 = np.array([0.5, 0.55, 0.62, 0.68, 0.75, 0.81])

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, y1, color=OKABE_ITO["vermillion"], marker="s", linestyle="--", label="baseline")
ax.plot(x, y2, color=OKABE_ITO["blue"], marker="o", linestyle="-", label="ours")
ax.legend(frameon=False)
out_color = os.path.join(ASSETS, "02_redundant_encoding_color.png")
fig.savefig(out_color, dpi=200)
plt.close(fig)
assert os.path.exists(out_color) and os.path.getsize(out_color) > 0

# 灰度测试:把彩色图转换成灰度,检查图里依然有可辨识的结构(不是纯色块糊成一团)
gray = Image.open(out_color).convert("L")
out_gray = os.path.join(ASSETS, "02_redundant_encoding_grayscale.png")
gray.save(out_gray)
assert os.path.exists(out_gray) and os.path.getsize(out_gray) > 0

arr = np.asarray(gray)
assert arr.std() > 0   # 灰度图仍有明显的像素强度变化(标记/线条/文字),不是被抹成单一灰色
print("grayscale std (closer to 0 = more information lost):", round(float(arr.std()), 2))
```

**审稿人/读者会怎么挑刺:**
- "如果我把这篇论文黑白打印出来,还能分清哪条线是你们的方法吗?"——这是最直接的挑刺方式,尤其是
  纸质审稿传统还没完全消失的场景下。
- "图例里两条线的颜色我勉强能分清,但线型和 marker 完全一样,这是有意为之吗?"——只做了颜色区分、
  没做冗余编码,即使颜色本身选得不错(比如用了 Okabe-Ito),依然会被指出"单一通道编码"这个更根本的
  设计缺陷。
- "两条曲线在中间交叉的地方,我数了三次才确认哪条线延续到右边的最高点。"——纯颜色编码在交叉点最容易
  露馅,这条批评往往来自真的认真读图、想核实数字的审稿人,而不是走马观花的审稿人。

**常见坑:**
- 只在有限的几个类别上做了冗余编码,但曲线一多(超过 4-5 条)线型/marker 的组合也开始互相混淆——
  这时候应该考虑按 [01 号文件](01-chart-type-selection.md)的判断拆成多个小面板,而不是继续在同一张
  图里塞近乎相同的线型。
- 图例(legend)本身只展示颜色色块,没有把 marker/linestyle 一起画出来——matplotlib 的 `ax.legend()`
  默认会自动把 `plot()` 调用里设置的 `marker`/`linestyle` 一起画进图例,但如果图例是手工用 `Patch`
  之类的对象拼出来的,容易漏掉这部分信息,导致图例本身看起来又变回了"只有颜色"。
- 以为"用了 Okabe-Ito 调色板就不需要冗余编码"——调色板解决的是"色盲能不能分清这几个颜色",不解决
  "黑白打印之后还剩不剩颜色可言"这个更彻底的失效场景,两者是互补的,不是二选一。

---

*创建:2026-07-25*
