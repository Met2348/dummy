# 07 · 手把手实战:画一张投稿级别的核心结果图

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 7 个"知识点",不计入本系列 22 个知识点的统计——
> 和 [dsa-deep-dive/21 号文件](../dsa-deep-dive/21-build-a-mini-search-engine.md)是同一挂,格式是
> "教程体":从一个空白想法开始,分阶段一步步敲代码,每写一段就在 `.venv` 里真跑一次、看到真实生成的
> 图片,最后拼成一张完整的、多面板的、投稿级别的核心结果图,而不是一次性甩出一大段成品代码。

## 为什么是"这张图"

前 6 篇([01](01-chart-type-selection.md) 图表类型、[02](02-color-and-perception.md) 颜色感知、
[03](03-multi-panel-layout-engineering.md) 多面板排版、[04](04-architecture-and-flow-diagrams.md)
架构图工具、[05](05-caption-writing.md) 图注写作、[06](06-anti-patterns.md) 反模式)各自讲的是一条
独立判断,这一篇要把它们**拼到同一张图上**,验证这些判断合起来能不能真的产出一张站得住的图:

| 阶段 | 要让这张图多具备一个能力 | 建立在哪篇已有内容之上 |
|------|------|------|
| 阶段 0 | 想清楚这张图要讲的"一句话故事" | 全篇的前提,不写代码 |
| 阶段 1 | 造出形状合理的数据,先不画图 | 是后面一切的地基 |
| 阶段 2 | 单面板折线图,带误差棒 + 冗余编码 | [01 类](01-chart-type-selection.md)折线图选型、[02 类](02-color-and-perception.md)Okabe-Ito 配色+冗余编码 |
| 阶段 3 | 拼成双面板复合图 | [03 类](03-multi-panel-layout-engineering.md) `gridspec` |
| 阶段 4 | 配上规范图注,导出投稿双格式 | [03 类](03-multi-panel-layout-engineering.md)矢量/位图、[05 类](05-caption-writing.md)图注自洽性 |
| 阶段 5 | 组装成一个完整的构建函数,跑一次端到端验证 | 阶段 1-4 全部组装 + [06 类](06-anti-patterns.md)跨面板一致性 |

每个阶段的代码都能独立运行(本文件用仓库统一的 [`_verify_md.py`](_verify_md.py) 校验,校验方式是把
每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以
后面阶段用到前面阶段的代码时会重新贴一遍,这是校验机制要求的,不是偷懒复制)。

**这张图画的是什么场景**:借用用户真实在研项目
[`research/world-model-imagination-controller/`](../../research/world-model-imagination-controller/)
(世界模型测试时想象预算自适应分配控制器,即将投稿 ICLR)的真实场景设定——该项目会前简报
[`01-meeting-briefing.md`](../../research/world-model-imagination-controller/01-meeting-briefing.md)
§3.4 的真实 pilot 发现是"想象和基线共享同一个不完美模型时,决策时多算几步只会引入噪声";二次深挖
[`02-deep-gap-analysis.md`](../../research/world-model-imagination-controller/02-deep-gap-analysis.md)
进一步确认"给想象真正的任务相关信息优势,才能让它从负翻正"是这个项目里唯一站得住"系统性新发现"的
一条。本教程照着这个**真实的论证结构**(想象预算 vs 表现,有没有信息优势决定想象划不划算)设计一张
"这类论文真会画出来"的核心结果图,但下面阶段 1 生成的具体数字是本教程自己的合成数据(有明确随机种子、
形状经过 sanity check),不是项目的真实实验结果——真实数字留给项目自己的论文用,教学案例只借场景,
这是设计文档"跨系列共同纪律"第 2 条的要求,[00-roadmap.md](00-roadmap.md)开头也已经声明过一次。

---

## 阶段 0:先想清楚"一句话故事",不写代码

动手画图之前,先回答[05 号文件第 2 条](05-caption-writing.md#2-陈述式标题-vs-描述式标题一图一个故事怎么落到文字)
的那个问题:**这张图读完之后,读者应该带走哪一句陈述式结论?**

这一篇选定的故事是:**"想象只有在获得任务相关信息优势时才划算——没有信息优势时,想象预算越高,
表现反而越差。"** 这句话直接决定了后面每一步的设计:
- 需要几条曲线?——至少三条:不想象的固定基线(表现应该和预算无关,一条平线)、"想象但没有任务
  信息优势"(呼应真实项目发现一,预算越高越吃亏)、"想象且有任务信息优势"(预算越高越好)。
- x 轴是什么?——想象预算,一个有序连续变量,按[01 号文件第 1 条](01-chart-type-selection.md)的
  判断,该用折线图,不是柱状图。
- 需要几个面板?——两个:一个展示"随预算变化的完整趋势"(主要证据),一个展示"在最大预算下的汇总
  对比"(浓缩成一眼能看懂的两根柱子,呼应故事的落点)。

**如果这一步跳过、直接打开编辑器写 `plt.plot(...)`,最容易出的问题**:画到一半才发现"这条曲线到底
想说明什么"没想清楚,不断临时加线、加面板,最后拼出一张信息过载、没有单一故事线的图——这正是
[06 号文件](06-anti-patterns.md)反复强调的"每一份装饰/每一个面板都要能回答'它在为哪个结论服务'"这条
原则,从设计的第一步就该贯彻,不是画完以后再回头精简。

---

## 阶段 1:造合成数据,先不画图

有了故事,先把支撑这个故事的数据结构造出来,并且在画任何图之前,用 `assert` 确认数据本身的形状符合
故事的预期——这是全系列"能算的地方真算"纪律在数据生成阶段的应用:不能等图画出来、人眼看着"好像对"
就直接采信,先用断言核对趋势方向。

```python
import numpy as np

def simulate_data():
    """三条曲线,跨 6 个预算取值、5 个模拟随机种子:
    - no_imagine: 不想象的固定基线,理论上和预算无关,应该稳定在 0.50 附近
    - naive: 想象但没有任务信息优势(呼应真实项目发现一),预算越高越吃亏
    - adaptive: 想象且有任务信息优势,预算越高越好
    """
    rng = np.random.default_rng(42)
    budgets = np.array([1, 2, 3, 5, 8, 13])
    n_seeds = 5

    def simulate(mean_fn, noise_scale):
        means, stds = [], []
        for b in budgets:
            vals = mean_fn(b) + rng.normal(0, noise_scale, size=n_seeds)
            vals = np.clip(vals, 0, 1)   # 成功率截断在 [0,1],呼应01号文件第2条讲过的截断坑
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    no_imagine = simulate(lambda b: 0.50, 0.02)
    naive = simulate(lambda b: 0.50 - 0.008 * b, 0.025)
    adaptive = simulate(lambda b: 0.50 + 0.024 * b, 0.03)
    return budgets, no_imagine, naive, adaptive


budgets, (no_mean, no_std), (naive_mean, naive_std), (adap_mean, adap_std) = simulate_data()

# sanity check: 数据形状必须先符合"一句话故事",再动手画图——不能反过来靠画出来的图说服自己
assert len(budgets) == len(no_mean) == len(naive_mean) == len(adap_mean) == 6
assert abs(no_mean.mean() - 0.50) < 0.03          # 固定基线确实稳定在 0.50 附近
assert naive_mean[-1] < naive_mean[0]              # naive 曲线确实随预算增加而下降
assert adap_mean[-1] > adap_mean[0]                # adaptive 曲线确实随预算增加而上升
assert adap_mean[-1] > no_mean.mean() + 0.15        # 最终 adaptive 明显甩开基线,故事的"落点"站得住
assert (no_std >= 0).all() and (naive_std >= 0).all() and (adap_std >= 0).all()  # 标准差不能是负数

print("budgets:", budgets)
print("no_imagine mean:", np.round(no_mean, 3))
print("naive       mean:", np.round(naive_mean, 3))
print("adaptive    mean:", np.round(adap_mean, 3))
print("stage1 data sanity check passed")
```

实测输出:`no_imagine` 稳定在 0.50 附近波动,`naive` 从约 0.496 一路降到约 0.40,`adaptive` 从约
0.52 爬升到约 0.82——三条曲线的形状精确对上阶段 0 定好的故事,才进入下一步画图。

---

## 阶段 2:单面板折线图——误差棒 + 冗余编码

先只画阶段 0 故事里最关键的那个面板:三条曲线随预算变化的趋势。这一步直接应用
[02 号文件](02-color-and-perception.md)的两条判断:用 Okabe-Ito 调色板取色,且颜色之外再叠加
marker 形状和线型两个冗余编码通道(色盲读者/黑白打印场景下依然能分清三条线)。

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"black": "#000000", "vermillion": "#D55E00", "blue": "#0072B2", "grey": "#999999"}


def simulate_data():
    rng = np.random.default_rng(42)
    budgets = np.array([1, 2, 3, 5, 8, 13])
    n_seeds = 5

    def simulate(mean_fn, noise_scale):
        means, stds = [], []
        for b in budgets:
            vals = mean_fn(b) + rng.normal(0, noise_scale, size=n_seeds)
            vals = np.clip(vals, 0, 1)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    no_imagine = simulate(lambda b: 0.50, 0.02)
    naive = simulate(lambda b: 0.50 - 0.008 * b, 0.025)
    adaptive = simulate(lambda b: 0.50 + 0.024 * b, 0.03)
    return budgets, no_imagine, naive, adaptive


budgets, (no_mean, no_std), (naive_mean, naive_std), (adap_mean, adap_std) = simulate_data()

fig, ax = plt.subplots(figsize=(4.8, 3.4))
# 三重冗余编码: 颜色(Okabe-Ito) + marker形状 + 线型,任何一个通道失效,另外两个还能分清三条线
ax.errorbar(budgets, no_mean, yerr=no_std, color=OKABE_ITO["grey"],
            marker="^", linestyle=":", capsize=3, label="no imagination (fixed)")
ax.errorbar(budgets, naive_mean, yerr=naive_std, color=OKABE_ITO["vermillion"],
            marker="s", linestyle="--", capsize=3, label="imagination, no task info")
ax.errorbar(budgets, adap_mean, yerr=adap_std, color=OKABE_ITO["blue"],
            marker="o", linestyle="-", capsize=3, label="adaptive, task-conditioned")
ax.set_xlabel("imagination budget (rollouts per decision)")
ax.set_ylabel("task success rate")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()

out = os.path.join(ASSETS, "07_stage2_single_panel.png")
fig.savefig(out, dpi=200)
plt.close(fig)

img = Image.open(out)
assert img.size == (round(4.8 * 200), round(3.4 * 200))
assert os.path.exists(out) and os.path.getsize(out) > 0
print("stage2 single-panel figure saved:", out, "size:", img.size)
```

---

## 阶段 3:`gridspec` 拼成双面板复合图

按阶段 0 的设计,补上第二个面板——把"最大预算下的汇总对比"浓缩成两根柱子,用
[03 号文件第 1 条](03-multi-panel-layout-engineering.md)的 `gridspec` 把两个面板拼进同一张图,
`width_ratios` 让主面板(趋势线)占更多版面,汇总面板占较少版面。

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"black": "#000000", "vermillion": "#D55E00", "blue": "#0072B2", "grey": "#999999"}


def simulate_data():
    rng = np.random.default_rng(42)
    budgets = np.array([1, 2, 3, 5, 8, 13])
    n_seeds = 5

    def simulate(mean_fn, noise_scale):
        means, stds = [], []
        for b in budgets:
            vals = mean_fn(b) + rng.normal(0, noise_scale, size=n_seeds)
            vals = np.clip(vals, 0, 1)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    no_imagine = simulate(lambda b: 0.50, 0.02)
    naive = simulate(lambda b: 0.50 - 0.008 * b, 0.025)
    adaptive = simulate(lambda b: 0.50 + 0.024 * b, 0.03)
    return budgets, no_imagine, naive, adaptive


budgets, (no_mean, no_std), (naive_mean, naive_std), (adap_mean, adap_std) = simulate_data()

# layout="constrained": 03号文件第1条讲过的现代布局引擎,避免多面板挤压/标签重叠
fig = plt.figure(figsize=(7.2, 3.2), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

axA = fig.add_subplot(gs[0, 0])
axA.errorbar(budgets, no_mean, yerr=no_std, color=OKABE_ITO["grey"],
             marker="^", linestyle=":", capsize=3, label="no imagination (fixed)")
axA.errorbar(budgets, naive_mean, yerr=naive_std, color=OKABE_ITO["vermillion"],
             marker="s", linestyle="--", capsize=3, label="imagination, no task info")
axA.errorbar(budgets, adap_mean, yerr=adap_std, color=OKABE_ITO["blue"],
             marker="o", linestyle="-", capsize=3, label="adaptive, task-conditioned")
axA.set_xlabel("imagination budget (rollouts per decision)")
axA.set_ylabel("task success rate")
axA.set_ylim(0.2, 0.95)
axA.legend(frameon=False, fontsize=7, loc="upper left")
axA.set_title("A", loc="left", fontweight="bold")

# panel B 的配色用和 panel A 完全相同的颜色字典驱动,避免06号文件第4条讲的"跨面板配色不一致"
COLOR_BY_METHOD = {"no info": OKABE_ITO["vermillion"], "task-\nconditioned": OKABE_ITO["blue"]}
axB = fig.add_subplot(gs[0, 1])
final_labels = ["no info", "task-\nconditioned"]
final_means = [naive_mean[-1], adap_mean[-1]]
final_stds = [naive_std[-1], adap_std[-1]]
axB.bar(final_labels, final_means, yerr=final_stds, capsize=4,
        color=[COLOR_BY_METHOD[l] for l in final_labels])
axB.axhline(no_mean.mean(), color=OKABE_ITO["black"], linewidth=0.8, linestyle=":")
axB.text(1.0, no_mean.mean() + 0.02, "no-imagination baseline", fontsize=6, ha="right")
axB.set_ylabel("success rate at budget=13")
axB.set_ylim(0.2, 0.95)
axB.set_title("B", loc="left", fontweight="bold")

fig.suptitle(
    "Imagination only pays off with a task-relevant information edge",
    fontsize=10, fontweight="bold", x=0.02, ha="left",
)

out = os.path.join(ASSETS, "07_stage3_two_panel.png")
fig.savefig(out, dpi=200)
plt.close(fig)

img = Image.open(out)
assert img.size == (round(7.2 * 200), round(3.2 * 200))
assert os.path.exists(out) and os.path.getsize(out) > 0
print("stage3 two-panel figure saved:", out, "size:", img.size)
```

---

## 阶段 4:规范图注 + 投稿双格式导出

按[05 号文件](05-caption-writing.md)的判断写图注:一句不引用具体面板的陈述式主标题(已经作为
`fig.suptitle` 画在图上了,图注里再完整重复一遍——图上的大标题是给"扫一眼图"的读者看的,图注文字
是给"认真读图"的读者看的,两者受众不同,内容重复是必要的,不是冗余),接上 `(A) ...`/`(B) ...` 的
逐面板说明,并且用[05 号文件第 1 条](05-caption-writing.md#1-自洽性原则caption-要能脱离正文被理解)
写的检查函数验证这条图注真的合格。导出格式按[03 号文件第 3 条](03-multi-panel-layout-engineering.md)
的判断,PDF(矢量,投稿正文用)和 PNG(300dpi 位图,预览/展示用)两份都要有。

```python
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"black": "#000000", "vermillion": "#D55E00", "blue": "#0072B2", "grey": "#999999"}


def simulate_data():
    rng = np.random.default_rng(42)
    budgets = np.array([1, 2, 3, 5, 8, 13])
    n_seeds = 5

    def simulate(mean_fn, noise_scale):
        means, stds = [], []
        for b in budgets:
            vals = mean_fn(b) + rng.normal(0, noise_scale, size=n_seeds)
            vals = np.clip(vals, 0, 1)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    no_imagine = simulate(lambda b: 0.50, 0.02)
    naive = simulate(lambda b: 0.50 - 0.008 * b, 0.025)
    adaptive = simulate(lambda b: 0.50 + 0.024 * b, 0.03)
    return budgets, no_imagine, naive, adaptive


def check_caption(caption, has_error_bars):
    """05号文件第1条定义的图注自洽性检查函数,原样复用。"""
    VAGUE_OPENERS = ["shows", "plot of", "graph of", "figure of", "illustration of"]
    issues = []
    lowered = caption.strip().lower()
    if any(lowered.startswith(o) for o in VAGUE_OPENERS):
        issues.append("opens with a purely descriptive phrase instead of a declarative finding")
    if has_error_bars and not re.search(
        r"error bar|standard (deviation|error)|\bci\b|confidence interval|\bstd\b", lowered
    ):
        issues.append("figure has error bars but caption never defines what they represent")
    if len(caption.split()) < 8:
        issues.append("too short to be self-contained (fewer than 8 words)")
    return issues


budgets, (no_mean, no_std), (naive_mean, naive_std), (adap_mean, adap_std) = simulate_data()

fig = plt.figure(figsize=(7.2, 3.2), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
axA = fig.add_subplot(gs[0, 0])
axA.errorbar(budgets, no_mean, yerr=no_std, color=OKABE_ITO["grey"], marker="^", linestyle=":", capsize=3, label="no imagination (fixed)")
axA.errorbar(budgets, naive_mean, yerr=naive_std, color=OKABE_ITO["vermillion"], marker="s", linestyle="--", capsize=3, label="imagination, no task info")
axA.errorbar(budgets, adap_mean, yerr=adap_std, color=OKABE_ITO["blue"], marker="o", linestyle="-", capsize=3, label="adaptive, task-conditioned")
axA.set_xlabel("imagination budget (rollouts per decision)")
axA.set_ylabel("task success rate")
axA.set_ylim(0.2, 0.95)
axA.legend(frameon=False, fontsize=7, loc="upper left")
axA.set_title("A", loc="left", fontweight="bold")

COLOR_BY_METHOD = {"no info": OKABE_ITO["vermillion"], "task-\nconditioned": OKABE_ITO["blue"]}
axB = fig.add_subplot(gs[0, 1])
final_labels = ["no info", "task-\nconditioned"]
axB.bar(final_labels, [naive_mean[-1], adap_mean[-1]], yerr=[naive_std[-1], adap_std[-1]],
        capsize=4, color=[COLOR_BY_METHOD[l] for l in final_labels])
axB.axhline(no_mean.mean(), color=OKABE_ITO["black"], linewidth=0.8, linestyle=":")
axB.text(1.0, no_mean.mean() + 0.02, "no-imagination baseline", fontsize=6, ha="right")
axB.set_ylabel("success rate at budget=13")
axB.set_ylim(0.2, 0.95)
axB.set_title("B", loc="left", fontweight="bold")
fig.suptitle("Imagination only pays off with a task-relevant information edge",
             fontsize=10, fontweight="bold", x=0.02, ha="left")

# 05号文件的图注规范: 陈述式主标题(不引用具体面板) + 逐面板说明 + 误差棒定义
caption = (
    "Imagination only pays off with a task-relevant information edge. "
    "(A) Task success rate as a function of imagination budget for three conditions; "
    "error bars show standard deviation across 5 seeds. "
    "(B) Aggregate success rate at the largest tested budget, same colors as (A); "
    "dotted line marks the no-imagination baseline."
)
caption_issues = check_caption(caption, has_error_bars=True)
assert caption_issues == []   # 图注真的通过了自洽性检查,不是写完就假设它合格

png_path = os.path.join(ASSETS, "07_capstone_final.png")
pdf_path = os.path.join(ASSETS, "07_capstone_final.pdf")
fig.savefig(png_path, dpi=300)   # 位图,预览/展示用
fig.savefig(pdf_path)            # 矢量,投稿正文用
plt.close(fig)

# 三层验证(见 00-roadmap.md"验证方法论差异声明"): 文件真的生成 + 格式元数据符合预期
assert os.path.exists(png_path) and os.path.getsize(png_path) > 0
assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
img = Image.open(png_path)
assert img.size == (round(7.2 * 300), round(3.2 * 300))
dpi_x, dpi_y = img.info["dpi"]
assert round(dpi_x) == 300 and round(dpi_y) == 300   # 用 round(),不是 =300——00号文件讲过精确等于300会失败
with open(pdf_path, "rb") as f:
    assert f.read(5) == b"%PDF-"

print("caption:", caption)
print("caption issues:", caption_issues)
print("png size:", img.size, "dpi:", img.info["dpi"])
print("stage4 caption + dual-format export verified")
```

---

## 阶段 5:组装成一个完整的构建函数,跑一次端到端验证

把阶段 1-4 拼进一个函数,加上[06 号文件第 4 条](06-anti-patterns.md#4-跨-panel-配色不一致多面板图必须共享同一套语义配色)
的跨面板一致性断言,跑一次完整的端到端验证。

```python
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors

ASSETS = r"E:\Workspace\dummy\for_real_dummy\research-figures-deep-dive\_assets"
os.makedirs(ASSETS, exist_ok=True)

OKABE_ITO = {"black": "#000000", "vermillion": "#D55E00", "blue": "#0072B2", "grey": "#999999"}


def simulate_data():
    rng = np.random.default_rng(42)
    budgets = np.array([1, 2, 3, 5, 8, 13])
    n_seeds = 5

    def simulate(mean_fn, noise_scale):
        means, stds = [], []
        for b in budgets:
            vals = mean_fn(b) + rng.normal(0, noise_scale, size=n_seeds)
            vals = np.clip(vals, 0, 1)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    no_imagine = simulate(lambda b: 0.50, 0.02)
    naive = simulate(lambda b: 0.50 - 0.008 * b, 0.025)
    adaptive = simulate(lambda b: 0.50 + 0.024 * b, 0.03)
    return budgets, no_imagine, naive, adaptive


def check_caption(caption, has_error_bars):
    VAGUE_OPENERS = ["shows", "plot of", "graph of", "figure of", "illustration of"]
    issues = []
    lowered = caption.strip().lower()
    if any(lowered.startswith(o) for o in VAGUE_OPENERS):
        issues.append("opens with a purely descriptive phrase instead of a declarative finding")
    if has_error_bars and not re.search(
        r"error bar|standard (deviation|error)|\bci\b|confidence interval|\bstd\b", lowered
    ):
        issues.append("figure has error bars but caption never defines what they represent")
    if len(caption.split()) < 8:
        issues.append("too short to be self-contained (fewer than 8 words)")
    return issues


def build_capstone_figure():
    """阶段1-4的完整组装:造数据 -> 双面板绘图(误差棒+冗余编码+gridspec) -> 图注 -> 双格式导出。
    返回文件路径 + 图注文本 + panel B 每根柱子的真实渲染颜色,供后面统一做端到端验证。"""
    budgets, (no_mean, no_std), (naive_mean, naive_std), (adap_mean, adap_std) = simulate_data()

    fig = plt.figure(figsize=(7.2, 3.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    axA = fig.add_subplot(gs[0, 0])
    axA.errorbar(budgets, no_mean, yerr=no_std, color=OKABE_ITO["grey"], marker="^", linestyle=":", capsize=3, label="no imagination (fixed)")
    axA.errorbar(budgets, naive_mean, yerr=naive_std, color=OKABE_ITO["vermillion"], marker="s", linestyle="--", capsize=3, label="imagination, no task info")
    axA.errorbar(budgets, adap_mean, yerr=adap_std, color=OKABE_ITO["blue"], marker="o", linestyle="-", capsize=3, label="adaptive, task-conditioned")
    axA.set_xlabel("imagination budget (rollouts per decision)")
    axA.set_ylabel("task success rate")
    axA.set_ylim(0.2, 0.95)
    axA.legend(frameon=False, fontsize=7, loc="upper left")
    axA.set_title("A", loc="left", fontweight="bold")

    COLOR_BY_METHOD = {"no info": OKABE_ITO["vermillion"], "task-\nconditioned": OKABE_ITO["blue"]}
    axB = fig.add_subplot(gs[0, 1])
    final_labels = ["no info", "task-\nconditioned"]
    bars = axB.bar(final_labels, [naive_mean[-1], adap_mean[-1]], yerr=[naive_std[-1], adap_std[-1]],
                    capsize=4, color=[COLOR_BY_METHOD[l] for l in final_labels])
    axB.axhline(no_mean.mean(), color=OKABE_ITO["black"], linewidth=0.8, linestyle=":")
    axB.text(1.0, no_mean.mean() + 0.02, "no-imagination baseline", fontsize=6, ha="right")
    axB.set_ylabel("success rate at budget=13")
    axB.set_ylim(0.2, 0.95)
    axB.set_title("B", loc="left", fontweight="bold")
    fig.suptitle("Imagination only pays off with a task-relevant information edge",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")

    caption = (
        "Imagination only pays off with a task-relevant information edge. "
        "(A) Task success rate as a function of imagination budget for three conditions; "
        "error bars show standard deviation across 5 seeds. "
        "(B) Aggregate success rate at the largest tested budget, same colors as (A); "
        "dotted line marks the no-imagination baseline."
    )

    png_path = os.path.join(ASSETS, "07_capstone_final.png")
    pdf_path = os.path.join(ASSETS, "07_capstone_final.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "png_path": png_path, "pdf_path": pdf_path, "caption": caption,
        "panel_b_colors": [b.get_facecolor() for b in bars],
    }


result = build_capstone_figure()

# 端到端验证清单,逐条对应本教程用到的每一篇前置知识
assert os.path.exists(result["png_path"]) and os.path.getsize(result["png_path"]) > 0   # 文件真的落盘
assert os.path.exists(result["pdf_path"]) and os.path.getsize(result["pdf_path"]) > 0

img = Image.open(result["png_path"])
assert img.size == (round(7.2 * 300), round(3.2 * 300))          # 03号文件: 像素尺寸=figsize*dpi
dpi_x, dpi_y = img.info["dpi"]
assert round(dpi_x) == 300 and round(dpi_y) == 300                # 03号文件: DPI元数据核对(容差,不是==300)

with open(result["pdf_path"], "rb") as f:
    assert f.read(5) == b"%PDF-"                                  # 03号文件: 矢量格式签名核对

assert check_caption(result["caption"], has_error_bars=True) == []  # 05号文件: 图注自洽性

expected_naive = mcolors.to_rgba(OKABE_ITO["vermillion"])
expected_adaptive = mcolors.to_rgba(OKABE_ITO["blue"])
assert result["panel_b_colors"][0] == expected_naive               # 06号文件: 跨面板配色一致性
assert result["panel_b_colors"][1] == expected_adaptive

print("PNG:", result["png_path"], img.size, img.info["dpi"])
print("PDF:", result["pdf_path"])
print("caption self-containment: passed")
print("cross-panel color consistency: passed")
print("END-TO-END CAPSTONE VERIFICATION: ALL PASSED")
```

到这里,`build_capstone_figure()` 已经是一个真实能用的小工具:换一组真实实验数据进去,就能直接产出
一张遵守本系列全部 6 条判断(图表类型选型、配色感知、多面板排版、图注规范、反模式规避)的投稿级图,
而且每一条判断都有对应的断言在守着,不是"画完之后自己觉得还不错"。

---

## 审稿人会怎么挑刺这张图(把前 6 篇的追问链集中应用一次)

这一节不是新知识点,是把 [01](01-chart-type-selection.md)-[06](06-anti-patterns.md) 每篇"审稿人/
读者会怎么挑刺"里提过的问题,集中套在这张最终图上自查一遍——这正是这张图设计时已经努力规避、但仍然
值得显式列出来的清单:

1. **误差棒代表什么、几个种子算出来的?**——已经在图注里写明"standard deviation across 5 seeds"
   ([05 号文件第 1 条](05-caption-writing.md))。
2. **色盲读者能分清三条线吗?黑白打印呢?**——颜色(Okabe-Ito)+ marker + 线型三重冗余编码
   ([02 号文件第 4 条](02-color-and-perception.md))。
3. **panel A 和 panel B 里同一个方法的颜色一致吗?**——由同一份 `COLOR_BY_METHOD` 字典驱动,阶段 5
   已经用 `get_facecolor()` 真实断言过一致性([06 号文件第 4 条](06-anti-patterns.md))。
4. **主标题有没有不小心引用某个具体面板?**——`fig.suptitle` 和图注开头那句都是"Imagination only
   pays off with a task-relevant information edge",没有出现"panel A"/"panel B"字样
   ([05 号文件第 3 条](05-caption-writing.md))。
5. **这张图导出的格式适合投稿吗?**——PDF(矢量)和 PNG(300dpi 位图)都导出了,分工明确
   ([03 号文件第 3 条](03-multi-panel-layout-engineering.md))。
6. **缩小到论文单栏宽度(约 3.3 英寸)之后还能看清楚吗?**——**这条本教程刻意没有做**:阶段 2-5 全部
   用的是 `figsize=(7.2, 3.2)`(双栏跨栏尺寸),不是单栏尺寸,这是有意示范"多面板复合图通常需要跨栏
   排版"这个真实场景,但也意味着如果这张图最终要塞进单栏,还需要按
   [03 号文件第 4 条](03-multi-panel-layout-engineering.md)"作者时尺寸陷阱"重新核对一遍缩小后的
   有效字号——这是一个诚实的范围声明,不是本教程的疏漏,是留给读者自己练习的下一步。

**这条清单本身也是一个可以推广的方法论**:任何一张准备投稿的图,画完之后都可以把系列里出现过的"审稿人
会怎么挑刺"逐条自查一遍,而不是等真正的审稿意见回来才第一次被问到这些问题。

---

## 这篇教程展示的方法论

和 [dsa-deep-dive/21 号文件](../dsa-deep-dive/21-build-a-mini-search-engine.md)一样,这篇教程验证的是
"一条已完成的深挖系列,能不能挑几个关联知识点、设计一个真实有用的产出物、分阶段增量实现"这个模式在
"图片产出物"而不是"纯逻辑工具"上是否依然成立。区别在于[dsa-deep-dive 的组装模式](../dsa-deep-dive/21-build-a-mini-search-engine.md)
拼的是"三个独立的算法能力"(倒排索引/Trie/堆),这里拼的是"六条独立的设计判断"(类型/颜色/排版/工具/
图注/反模式)——前者组装完是一个新的行为能力(能搜索),后者组装完是同一张图上六重同时生效的质量
保证,组装的"产物形态"不同,但"挑关联点 → 设计有价值的目标 → 分阶段增量验证"这个方法论内核是一致的。
本系列的验证方式全程遵守 [00-roadmap.md](00-roadmap.md)"验证方法论差异声明"——文件生成 + 格式元数据
+ 能算的设计判断真算,不假装能对图片内容做像素级 assert。

---

*创建:2026-07-25*
