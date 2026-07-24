# 01 · 图表类型选择深挖(Chart Type Selection)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇回答系列第一个、也是最容易被忽视的问题:**动手画图之前,
> 先决定要不要画图、画哪种图**。后面几篇(颜色、排版、图注、反模式)全部建立在"图表类型选对了"这个
> 前提之上——类型选错了,后面配色再漂亮、排版再工整,也是在讲一个错误的故事。

**本篇统一结构(七步,见 [00-roadmap.md](00-roadmap.md) 完整声明):** 签名/是什么 → 一句话 → 底层机制
(这里是视觉感知原理)→ AI 研究场景 → 可运行例子(真在 `.venv` 里跑过,产出真实图片)→ 审稿人/读者会
怎么挑刺 → 常见坑。

---

## 1. 变量类型决定图表类型——第一性原理,不是背对照表

**是什么:**
```
choose_chart_type(n_continuous_vars, n_categorical_vars, x_is_ordered, precision_matters) -> str
# 输入"数据的形状"描述,输出该用的图表类型名——判断规则本身见下面"可运行例子"完整定义
```

**一句话:** 图表类型不是审美选择,是数据结构的直接推论——先问"我有几个连续变量、几个类别变量、自变量
排不排得出先后顺序",答案几乎唯一确定该用什么图,这条判断规则甚至可以写成一个纯函数。

**底层机制/为什么这样设计(这里是感知原理):** 人眼对不同视觉编码通道的"精度判断能力"是有排序的——
Cleveland & McGill 1984 年的经典感知实验(图表设计教材反复引用的基础结论)证明,人眼判断"长度/位置"
的精度远高于判断"角度/面积",这也是[06 号文件](06-anti-patterns.md)要单独批判饼图和 3D 图的感知学
根源。落到具体选型上:
- **折线图**把"趋势"编码成连续的位置变化,只有在 x 轴本身有**顺序**(时间、预算、训练步数……)时才
  有意义——连接两个点的线段,视觉上隐含"中间存在过渡"这个语义,x 轴是无序类别时连线是在撒谎。
- **柱状图**把数值编码成长度(人眼最擅长精确比较的通道),适合无序类别之间的比较,但**代价是会把 x 轴
  强制按类别等距排列**——下面可运行例子会现场展示这个副作用。
- **散点图**用于两个连续变量之间没有预设的自变量/因变量方向时,单纯呈现"这两个东西一起变化时长什么
  样",不强加趋势线。
- **热力图**是"两个类别轴 × 一个连续值"这种矩阵型数据的唯一自然选择——比如超参数网格搜索结果(下面
  第 3 条会展开)。

**AI 研究场景:** 写 ICLR/NeurIPS 论文的 Method/Experiments 部分时,这条判断最常见的应用场景是"性能
vs 某个连续超参数(想象预算、训练步数、模型规模)"该画折线图还是柱状图——如果这个超参数在语义上是
"越大越贵、可以取任意接近的中间值"(比如想象预算 1/2/3/5/8/13),用折线图;如果是若干个互相独立、不
存在"之间"这个概念的方法名/消融配置("no imagination" / "always imagine" / "adaptive"),用柱状图。

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

budget = np.array([1, 2, 3, 5, 8, 13])   # 想象预算,故意用不均匀间隔(接近斐波那契数列)
success = np.array([0.50, 0.55, 0.58, 0.66, 0.72, 0.80])

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].plot(budget, success, color=OKABE_ITO_BLUE, marker="o")
axes[0].set_title("line: trend over ordered x", fontsize=9)
axes[1].bar([str(b) for b in budget], success, color=OKABE_ITO_BLUE)
axes[1].set_title("bar: treats x as categories", fontsize=9)
axes[2].scatter(budget, success, color=OKABE_ITO_BLUE)
axes[2].set_title("scatter: no connecting trend", fontsize=9)
for ax in axes:
    ax.set_ylim(0.4, 0.9)
fig.tight_layout()
out = os.path.join(ASSETS, "01_chart_type_comparison.png")
fig.savefig(out, dpi=200)
plt.close(fig)

# 验证方式见 00-roadmap.md"验证方法论差异声明":文件真的生成、尺寸符合 figsize*dpi 换算
img = Image.open(out)
assert img.size == (round(9 * 200), round(3 * 200))
assert os.path.getsize(out) > 0

# 决策函数本身的行为可以精确 assert(这条属于"能算的一定真算")
def choose_chart_type(n_continuous_vars, n_categorical_vars, x_is_ordered, precision_matters):
    if precision_matters and (n_continuous_vars + n_categorical_vars) <= 6:
        return "table"
    if n_continuous_vars == 2 and n_categorical_vars == 0:
        return "scatter"
    if n_continuous_vars == 1 and n_categorical_vars == 1 and x_is_ordered:
        return "line"
    if n_continuous_vars == 1 and n_categorical_vars == 1 and not x_is_ordered:
        return "bar"
    if n_continuous_vars == 1 and n_categorical_vars == 2:
        return "heatmap"
    return "unclear-default-to-table"

assert choose_chart_type(1, 1, x_is_ordered=True, precision_matters=False) == "line"
assert choose_chart_type(1, 1, x_is_ordered=False, precision_matters=False) == "bar"
assert choose_chart_type(2, 0, x_is_ordered=False, precision_matters=False) == "scatter"
assert choose_chart_type(1, 2, x_is_ordered=False, precision_matters=False) == "heatmap"
print("chart type comparison saved:", out, "size:", img.size)
```

**这个例子里一个没预设、画出来才看清楚的副作用**:中间的柱状图把 `budget=[1,2,3,5,8,13]` 强制按类别
等距排列在 x 轴上——看图会以为从 8 到 13 和从 1 到 2 的"距离"一样,但真实预算值相差 5 和相差 1,完全
不是一回事;左边的折线图和右边的散点图因为 x 轴是真正的数值轴,8 到 13 之间的间距天然比 1 到 2 之间宽,
诚实保留了这个信息。**这不是"柱状图有 bug",是柱状图这个图表类型的设计前提就是"x 轴是无序类别"**,
用在一个其实有序、且间距不均匀的连续变量上,间距信息就被丢弃了——这正是"图表类型要匹配变量类型"这条
判断的具体代价。

**审稿人/读者会怎么挑刺:**
- "你这张图为什么用折线图连接这几个点?这几个配置之间真的存在'之间的状态'吗,还是只是几个离散的
  消融实验?"——如果 x 轴其实是无序的方法名/消融配置,被审稿人这样问基本等于承认画错了图。
- "这个柱状图的 x 轴刻度看起来间隔均匀,但下面写的数值明明不是等差数列,这是不是想造成一种'变化很
  平滑'的错觉?"——柱状图强制等距排列类别这件事,如果类别本身有数值含义(比如预算 1/2/3/5/8/13),
  审稿人有理由怀疑这是不是故意选的图表类型来掩盖"高预算区间数据点稀疏"这个事实。
- "两个连续变量的散点图,为什么没有画出拟合线/置信区间?你想让读者自己脑补相关性吗?"——纯散点图如果
  本身就是想呈现相关趋势,读者会预期至少有一条趋势线或者一个相关系数标注。

**常见坑:**
- 用折线图连接类别本身无序的柱状数据(比如把几个不同架构的消融结果按柱状图的顺序连成折线),隐含了
  "从 A 到 B 存在过渡"这个不存在的语义。
- 柱状图的 x 轴等距排列副作用,在自变量数值分布本身不均匀(指数增长的预算、对数分布的学习率)时最容易
  被忽略,画图的人自己心里知道"数值不是均匀的",但图会诚实地把这条信息藏起来,读者除非去看坐标轴标签
  文字,否则不会意识到。

---

## 2. 折线图的多序列画法与误差带——`errorbar`

**是什么:**
```
ax.errorbar(x, y, yerr=std, color=..., marker=..., linestyle=..., capsize=...)
# yerr: 每个点的误差范围(通常是跨随机种子的标准差,或 95% 置信区间半宽)
# capsize: 误差棒两端"帽子"的长度(单位 points),capsize=0 时帽子完全不可见,容易被误认为没画误差棒
```

**一句话:** 多条曲线对比时,`errorbar` 把"跨种子/跨试验的不确定性"直接画进图里,是 NeurIPS 等会议
"论文检查清单"明确要求的东西——检查清单原话(2026 年版沿用 2021 年确立的条款)要求"结果是否配有误差棒、
置信区间,或其他形式的统计显著性信息,至少对支撑核心论点的实验成立",没有误差棒的对比曲线,审稿人有
充分理由怀疑这条曲线的"提升"是不是噪声。

**底层机制/为什么这样设计:** 误差棒的宽度不是随便定的,必须能回答"这个误差在衡量什么随机性来源"这个
问题——是跨随机种子?跨 train/test 划分?跨初始化?NeurIPS 检查清单明确要求论文正文说清楚这一点,而
不只是画出误差棒完事。另一个容易被忽视的细节:如果底层数据分布是非对称的(比如成功率被截断在 `[0,1]`
区间内,均值靠近 1 时),对称误差棒会画出"上界超过 1.0"或"下界小于 0"这种物理上不可能的区间——检查
清单原话专门点名过这个坑,画图前要么用非对称误差棒,要么确认均值离边界够远不会露馅。

**AI 研究场景:** 想象预算(或任何"多花算力换性能"类型的超参数)扫描曲线,几乎是这类论文的标配图——
横轴是预算,纵轴是任务表现,画两三条方法的曲线对比,谁的曲线在同样预算下更高、谁的曲线爬升更快,是
Method 部分最核心的证据。误差带/误差棒决定了这条曲线的说服力:一条没有误差棒的"更优"曲线,审稿人无法
判断这个差距是真实效应还是种子方差,这也是[真实调研发现的一个案例](00-roadmap.md):有研究者专门做过
实验,发现"移除 baseline 对比"能让审稿打分从 7 分掉到 4 分——没有严谨的对比证据,论点的可信度会被
显著打折扣,误差棒是这类证据里最基础的一层。

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

budget = np.array([1, 2, 3, 5, 8, 13])
baseline_mean = np.full(6, 0.50)
baseline_std = np.full(6, 0.03)
ours_mean = np.array([0.51, 0.55, 0.60, 0.68, 0.75, 0.81])
ours_std = np.array([0.04, 0.04, 0.05, 0.04, 0.05, 0.05])

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.errorbar(budget, baseline_mean, yerr=baseline_std, color=OKABE_ITO["vermillion"],
            marker="s", linestyle="--", capsize=3, label="fixed baseline")
ax.errorbar(budget, ours_mean, yerr=ours_std, color=OKABE_ITO["blue"],
            marker="o", linestyle="-", capsize=3, label="adaptive controller")
ax.set_xlabel("imagination budget")
ax.set_ylabel("task success rate")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = os.path.join(ASSETS, "01_line_errorbar.png")
fig.savefig(out, dpi=200)
plt.close(fig)

img = Image.open(out)
assert img.size == (round(4.5 * 200), round(3.2 * 200))
assert os.path.getsize(out) > 0

# 非对称截断的坑:如果均值离 1.0 太近,对称误差棒会画出不可能的区间,这里现场算一遍演示怎么发现
near_ceiling_mean = 0.97
near_ceiling_std = 0.05
upper = near_ceiling_mean + near_ceiling_std
assert upper > 1.0   # 97% +- 5% 的上界是 102%,任务成功率不可能超过 100%——对称误差棒在这里已经说谎
print("errorbar chart saved:", out, "size:", img.size)
print("near-ceiling upper bound (impossible):", upper)
```

**审稿人/读者会怎么挑刺:**
- "误差棒代表什么?标准差、标准误,还是 95% 置信区间?几个随机种子算出来的?"——NeurIPS 检查清单
  明确要求正文写清楚这三件事,图上不写、正文也不写,是可以被直接扣分的缺口。
- "你的方法在预算=8 时误差棒和 baseline 的误差棒有重叠,你怎么保证这个'提升'不是噪声?"——误差棒
  重叠区域大,是审稿人最容易抓到的"证据不够硬"的把柄,视觉上重叠通常意味着差距没有统计显著性。
- "为什么这条曲线的误差棒上界超过了纵轴的物理上限?"——上面例子现场复现了这个坑,一旦被审稿人截图
  发出来,是很难挽回的印象分。

**常见坑:**
- `capsize=0`(matplotlib 默认值)会让误差棒的"帽子"完全不可见,只剩一条细线,很容易被读者(甚至
  审稿人)直接看漏,误以为完全没有画误差棒——显式设置 `capsize=3` 左右能大幅提升可辨识度。
- 均值接近数据的物理边界(0 或 1、或者任何截断范围)时,对称误差棒可能画出超出物理意义的区间,上面
  例子已经现场算出这个数字,画图前应该检查 `mean ± std` 是否仍在合法范围内。
- 图例只用颜色区分两条线,色盲读者可能无法分辨——这里已经额外用 `marker`(圆点 vs 方块)和
  `linestyle`(实线 vs 虚线)做了冗余编码,[02 号文件第 4 条](02-color-and-perception.md)会展开讲
  这个原则本身。

---

## 3. 热力图——两个离散轴 × 一个连续值的矩阵可视化

**是什么:**
```
im = ax.imshow(grid, cmap="viridis", vmin=..., vmax=..., aspect="auto")
fig.colorbar(im, ax=ax, label="...")
# grid: 形状 (n_rows, n_cols) 的二维数组,每个格子是一个连续值
```

**一句话:** 当数据天然是"两个类别轴(或两个离散取值的超参数)交叉出的网格,每个格子一个连续数值"时,
热力图是唯一能一眼看出"整个网格哪个区域最好"的图表类型——把同样的数据拆成好几条折线来看,读者要在
脑子里做拼图,热力图直接把拼图结果画出来。

**底层机制/为什么这样设计:** 热力图本质是把"数值"编码成"颜色",这是人眼**精度最低**的编码通道之一
(不像折线图的位置编码那么精确)——这不是热力图的缺点,而是它的定位本身就不是"精确读数",是"一眼扫过
去找到整体格局/异常区域",精确数值该配一张表格或者在格子里叠加数字标注。这也是为什么热力图必须严格
搭配色盲友好且**感知均匀**的连续 colormap(默认的 `viridis`)——[02 号文件第 2 条](02-color-and-perception.md)
会用真实计算出的亮度序列证明为什么 `jet` 不能用在这里:亮度非单调的 colormap 会在数据里制造出实际不
存在的"假等高线"边界,这对热力图是致命的,因为热力图的全部信息量都压缩在颜色这一个通道里,没有位置/
长度这些更精确的通道做冗余备份。

**AI 研究场景:** 超参数网格搜索结果(比如"想象深度 H" × "候选数 K"两个维度各扫几个值,每个格子是
一个任务成功率)是这类图最典型的场景——一张热力图能同时呈现"H 和 K 两个维度分别怎么影响表现"以及
"是否存在某个 H、K 的组合好到不成比例"这种单独看边际曲线看不出来的交互效应。混淆矩阵(分类任务的
预测类别 × 真实类别)是另一个经典场景,只是这里的"连续值"通常是计数或归一化后的比例。

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

depths = [1, 2, 3, 5, 8]
candidates = [1, 3, 5, 10]
rng = np.random.default_rng(7)
grid = rng.uniform(0.3, 0.85, size=(len(candidates), len(depths)))

fig, ax = plt.subplots(figsize=(4.5, 3.2))
im = ax.imshow(grid, cmap="viridis", vmin=0.3, vmax=0.85, aspect="auto")
ax.set_xticks(range(len(depths)), depths)
ax.set_yticks(range(len(candidates)), candidates)
ax.set_xlabel("rollout depth H")
ax.set_ylabel("candidate count K")
fig.colorbar(im, ax=ax, label="task success rate")
fig.tight_layout()
out = os.path.join(ASSETS, "01_heatmap.png")
fig.savefig(out, dpi=200)
plt.close(fig)

img = Image.open(out)
assert img.size == (round(4.5 * 200), round(3.2 * 200))
assert os.path.getsize(out) > 0

# 热力图矩阵形状本身要能被 assert:行数=候选数取值个数,列数=深度取值个数
assert grid.shape == (len(candidates), len(depths)) == (4, 5)
print("heatmap saved:", out, "grid shape:", grid.shape)
```

**审稿人/读者会怎么挑刺:**
- "颜色刻度(colorbar)的范围是从数据的最小值最大值自动定的,还是手动设的?如果几张热力图之间颜色
  刻度范围不一样,怎么互相比较?"——多张热力图并排放在一篇论文里,如果各自用各自的 `vmin`/`vmax`
  自动定刻度,颜色看起来一样深浅但实际数值可能差很多,这是审稿人经常抓的一个眼熟的坑。
- "这张热力图用的是 `jet` 还是感知均匀的 colormap?"——[02 号文件](02-color-and-perception.md)会
  展开讲这条,但热力图是这条批评最常命中的图表类型,因为它的信息量 100% 压在颜色通道上。
- "格子里能不能标出具体数值?只看颜色我没法判断这两个相邻格子到底差多少。"——如果论文的核心论点
  依赖某两个格子之间的精确差距,光有热力图不够,得叠加数字标注或者额外配一张表。

**常见坑:**
- 多张热力图并排比较时忘记统一 `vmin`/`vmax`,导致"看起来最好的格子"其实是色阶自动拉伸出来的视觉
  错觉,不是真的数值最高——上面例子里显式传入了固定的 `vmin=0.3, vmax=0.85`,这是有意为之,不是
  随手写的参数。
- `aspect="auto"`不设置的话,`imshow` 默认会强制格子是正方形(`aspect="equal"`),当行数和列数差很多
  时会把图挤压变形,`aspect="auto"` 让格子填满整个坐标轴区域,更适合网格搜索这种行列语义不同的场景。
- 忘记检查 `grid` 数组的行列语义和 `set_xticks`/`set_yticks` 标注的对应关系是否搞反——`imshow` 默认
  第一个维度(行)对应纵轴,第二个维度(列)对应横轴,如果构造 `grid` 时习惯性写成 `grid[depth_idx, candidate_idx]`,
  会和这里的 `grid[candidate_idx, depth_idx]` 刚好转置,标注全部对不上。

---

## 4. Table vs Figure 怎么选——精度重要用表,趋势重要用图

**是什么:**
```
table_or_figure(n_data_points, exact_values_matter, want_trend_or_pattern) -> str
# 判断规则本身见下面"可运行例子"完整定义
```

**一句话:** 数值本身的精确大小是论点的关键(比如要逐项列出消融实验的具体百分比供读者自己核对),用
表格;读者需要看出的是"趋势/比较/模式"而不是某个具体数字,用图——两者选错最常见的后果不是"难看",
是"读者要么找不到精确数字,要么在一堆数字里看不出模式"。

**底层机制/为什么这样设计:** 表格和图表其实是同一份数据的两种不同"压缩方式"。表格几乎不压缩信息
(每个数字原样列出,读者自己在脑子里比较),适合"数字本身就是重点、类别/变量很多、且都遵循同一种
格式"的场景;图表把数字压缩成位置/长度/颜色等视觉属性,牺牲了精确可读性,换来"一眼看出趋势"的效率。
这也是为什么"能一句话说清楚就不需要图表"——如果最终结论只有一句话("A 比 B 好"),那么无论表格还是
图表都是在为一个已经不需要视觉压缩的结论做多余的包装;只有当结论本身依赖"看出一个模式"时,压缩成
视觉才有意义。同一份数据不应该同时用表格和图表各展示一遍——那是同一件事说了两遍,不是两个论点。

**AI 研究场景:** 论文的消融实验(ablation)表,通常每一行是一个配置、每一列是一个指标,数值本身
(精确到小数点后一两位)就是论点的一部分,适合表格;而"性能随预算/规模变化"的扫描结果,读者真正关心
的是"曲线的形状"(单调递增?饱和?),适合图。一个常见的真实判断失误是把只有 2-3 行、2-3 列的小表格
硬要画成柱状图凑一张"图"——这类"迷你表格"级别的数据,无论画成图还是留作表格,都不如直接写进正文
一句话("我们的方法比最强 baseline 高 4.2 个百分点")来得高效。

**可运行例子:**
```python
def table_or_figure(n_data_points, exact_values_matter, want_trend_or_pattern):
    if n_data_points <= 2:
        return "neither -- just say it in one sentence"
    if exact_values_matter and not want_trend_or_pattern:
        return "table"
    if want_trend_or_pattern:
        return "figure"
    return "table"

# 只有一个对比数字(我们 vs 最强 baseline)——一句话就够,不需要图表
assert table_or_figure(1, exact_values_matter=True, want_trend_or_pattern=False) == "neither -- just say it in one sentence"

# 20 行消融实验,每个数字都要能被读者核对——表格
assert table_or_figure(20, exact_values_matter=True, want_trend_or_pattern=False) == "table"

# 20 个预算取值,关心的是曲线形状不是精确读数——图
assert table_or_figure(20, exact_values_matter=False, want_trend_or_pattern=True) == "figure"

# 数据点不算多,但既要看出趋势又想让读者能核对具体数——图(位置编码本身可以标数值标注,表格没有"形状")
assert table_or_figure(6, exact_values_matter=True, want_trend_or_pattern=True) == "figure"

print("table-vs-figure decision fn verified on 4 scenarios")
```

**审稿人/读者会怎么挑刺:**
- "这张图里的数据其实只有 6 个点,直接列成一句话或者一个小表格不是更清楚吗?为什么要占一整张图的
  篇幅?"——过度包装小数据是常见的"为了有图而画图",审稿人对页数有限的会议论文尤其敏感。
- "正文表格 3 和图 4 画的是同一组数据,为什么要重复展示两遍?"——同一份数据既上表又上图,除非表格
  聚焦精确数值、图表聚焦趋势形状(且正文明确说明两者的分工),否则是纯粹的冗余,浪费宝贵的页面篇幅。
- "这张表格有 15 列,我需要来回数才能找到我关心的那一列,能不能按重要性重新排列或者拆成两张?"——
  表格虽然精度高,但列数一多同样有可读性问题,不是"数字多就无脑上表格"就万事大吉。

**常见坑:**
- 把"只有两三个数字的对比"硬画成图表(比如只有 baseline 和 ours 两根柱子的柱状图),图表带来的视觉
  开销(占用版面、读者要重新定位坐标轴)超过了它节省的阅读成本,这种规模的对比几乎总是一句话文字更
  高效。
- 表格和图表内容重复(同一组数字既列表又画图),违反"不要在正文里把同一件事说两遍"的基本写作纪律,
  也是版面浪费。
- 表格试图塞下太多列/太多小数位精度,反而丧失了表格本该有的"精确、易于查找"的优势——真正需要极高
  精度的数字,应该放进 Supplementary,正文表格只保留读者真正会用来做判断的那几列。

---

*创建:2026-07-25*
