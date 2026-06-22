# L2 · matplotlib 出版级实操: 从「能看」到「能印」

> 35-min lecture · 目标: 把 L1 的判断力落实成代码 —— 用 rcParams / 字号 / 矢量格式 / 误差棒, 把默认 matplotlib 图升级到顶会论文水准, 并固化成可复用的样式包。

---

## 0. 为什么默认图不能直接进论文

你 `plt.plot(...)` 出来的默认图, 放进论文会出三类问题:

1. **缩小后看不清**: 论文图会被排版缩到**单栏宽 (~3.3 inch)**。默认字号在缩小后小得读不出。
2. **位图模糊**: 默认存 PNG, 放大/打印会马赛克。论文要**矢量 PDF**。
3. **chartjunk**: 默认带上/右边框、有时有网格, data-ink 偏低 (L1)。

> 解决思路: **不要每张图手调**, 而是把「出版级规范」固化成一次性设置 (rcParams) + 一个样式包 (`plotstyle.py`), 以后 import 即用。这既保证一致性, 又省时间。这正是本专题最实用的产出。

---

## 1. 三个最关键的旋钮

### ① 字号 (最重要)

论文图缩到单栏宽后, 图里字号要相当于正文的 **7-9pt** 才读得清。在大图上画时反而要设**大**字号, 缩小后才合适。

```python
mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
```

> 自检: 把图导出, 缩放到论文里的实际大小 (单栏 ~3.3in 宽), 字还看得清吗? 看不清 → 加大字号或减少图里的文字。**审稿人最常抱怨的就是「图里的字太小」。**

### ② 图尺寸 (匹配栏宽)

按论文栏宽定 figsize, 别随手 `figsize=(10,6)` 然后被排版压扁变形:
- **单栏图**: `figsize=(3.3, 2.4)` (本专题 `column_figsize()`)
- **双栏通栏图**: `figsize=(6.9, 2.6)`

按目标尺寸画, 字号和线宽的比例才正确。

### ③ 矢量格式 (PDF/SVG, 不是 PNG)

```python
fig.savefig("fig.pdf", bbox_inches="tight")   # 投稿用矢量
fig.savefig("fig.png", dpi=300)               # 预览/slides 用位图
```

> **矢量 (PDF/SVG)**: 由线条和文字定义, 无限缩放不糊, 是 LaTeX 投稿标准。**位图 (PNG/JPG)**: 由像素定义, 放大马赛克。规则: **线条图 (柱/线/示意图) 一律 PDF; 含大量像素的 (如热力图、真实图片) 可 PNG 高 DPI。** 本专题 `save_figure` 同时出 PDF + PNG。

---

## 2. data-ink 的代码落地: 去 chartjunk

把 L1 的「删墨水」变成几行:

```python
ax.spines[["top", "right"]].set_visible(False)   # 去上/右边框
ax.grid(False)                                    # 或 ax.grid(alpha=0.3) 极淡
# 不要 3D、不要背景色、不要柱子渐变
```

`plotstyle.set_pub_style()` 把这些默认打开 (`axes.spines.top/right = False`), 你画图时就不用每次记。

---

## 3. 误差棒: 让图诚实 (接 9.4-L5 / 9.5)

任何含均值的图都必须带不确定性。这是 L1 说的铁律, 也是 9.4-L5 的延续:

```python
ax.bar(x, means, yerr=sems, capsize=4)            # 柱状图误差棒
ax.errorbar(x, means, yerr=sems, fmt="o-", capsize=4)  # 折线图误差棒
# 或用阴影带 (折线常用):
ax.fill_between(x, means - sems, means + sems, alpha=0.2)
```

> **务必在图注里写清误差棒是什么** (std? SEM? 95% CI? 几个种子?)。不写, 审稿人无法判断, 还会怀疑你藏东西 (9.3 攻击清单 Q3/Q9 的可视化版)。`plotstyle.grouped_bar` 直接吃 9.4/9.5 的 mean/std。

---

## 4. 子图与布局: 多图协同

一篇论文常把相关的几张图并排成一个 figure (如 (a) 主结果 (b) 消融 (c) 敏感性):

```python
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), constrained_layout=True)
for ax, (title, data) in zip(axes, panels):
    ...
    ax.set_title(f"({chr(97+i)}) {title}")   # (a) (b) (c)
```

要点:
- `constrained_layout=True` 或 `fig.tight_layout()` 防止子图标签互相重叠。
- 子图用 **(a)(b)(c)** 编号, 正文和图注用编号引用。
- **共享坐标轴**时 (`sharey=True`) 只在最左标 y 轴, 省墨水 (data-ink)。

---

## 5. 一个可复用样式包的价值

把上面所有规范固化进 `plotstyle.py`:
- `set_pub_style()`: 一行设好字号/线宽/去边框/色盲色环/矢量 DPI。
- `column_figsize()`: 给正确的栏宽尺寸。
- `grouped_bar()` / `save_figure()`: 带误差棒的标准图 + 双格式导出。

> 为什么固化成包而不是每次手调: ① **一致性** —— 你论文里所有图风格统一 (审稿人觉得专业); ② **效率** —— 下次画图 `set_pub_style()` 一行搞定; ③ **可复现** —— 图也是代码产物 (9.5), 改数据重跑就重生成, 不用手动重画。**这个 80 行的包, 你读真博士时能一直用下去。**

---

## 6. 本讲小结 + 通往 L3

- 默认图三宗罪: 缩小看不清 (字号)、位图糊 (格式)、chartjunk (data-ink)。
- 三关键旋钮: **字号** (缩小后 7-9pt) / **图尺寸** (匹配栏宽) / **矢量格式** (线条图用 PDF)。
- 去 chartjunk: 去上右边框、去网格/3D; 含均值必带**误差棒** + 图注写清它是什么。
- 多图用子图 + (a)(b)(c) 编号 + constrained_layout。
- 把规范固化成可复用 `plotstyle` 包: 一致、高效、可复现。

> **下一讲 L3「方法示意图」**: 前面都是**数据图**。但论文的 Figure 1 通常是**方法示意图** (pipeline/架构), 它讲的不是数据而是「你的方法怎么运作」。L3 教你设计清晰的方法图 —— 这是审稿人理解你贡献的第一道门。

**动手**: 去 `N1-publication-quality-plot.ipynb`, 拿 9.4 的交互效应数据, 从「默认 matplotlib 图」一步步升级到「出版级」(set_pub_style → 误差棒 → 去 chartjunk → 导出 PDF), 做一个 before/after 对比。
