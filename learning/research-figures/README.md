# 9.6 research-figures — 科研绘图: 从「能看」到「能印进顶会」

> **Module 9「科研技能」· 阶段: 输出 (output)**
> 实验设计对了 (9.4)、可靠跑出并留痕了 (9.5), 接下来把结果**讲给人看**。本专题教你做出版级科研图 —— 既清晰有力, 又诚实经得起审稿人攻击。

---

## 这个专题要解决的真问题

图是你论文的**橱窗和论点**。一个审 6 篇的累审稿人, 常常看完 Figure 1 (方法) 和主结果图就形成了接受/拒绝的初判。但新手的图通常:

- 字太小 (缩到单栏宽看不清)、存成模糊 PNG、带一堆 chartjunk (3D/网格/边框)
- 没有误差棒 (把有方差的均值画成确定值)
- y 轴偷偷截断, 把 0.3% 吹成「巨大提升」(无意的, 但砸可信度)
- 方法图一团乱麻, 审稿人 30 秒看不懂你在做什么

> **会画图 = 让审稿人秒懂你的贡献, 且每张图都经得起你 9.3 学的那套攻击。** 这一层薄但关键: 同样的工作, 好图能让它被高估, 烂图能让它被埋没。

```
   9.4 数据 → 9.5 留痕 → 9.6: L1判断力 → L2出版级 → L3方法图 → L4诚实
                            (一图一信息) (rcParams/矢量/误差棒) (Figure 1) (不截断/图注)
                                        → 一张能进顶会、又诚实的图
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-grammar-of-a-good-figure.md` | 一张好图的语法: 一图一信息 / data-ink / 图类型 / 色彩 | 图的判断力 |
| L2 | `lectures/L2-matplotlib-publication.md` | matplotlib 出版级: rcParams / 字号 / 矢量 / 误差棒 | 可复用样式包 |
| L3 | `lectures/L3-schematic-diagrams.md` | 方法示意图: 论文 Figure 1 的设计 | 一张方法图 |
| L4 | `lectures/L4-figure-honesty.md` | 图的诚实性 + self-contained 图注 + 与正文呼应 | 诚实图 + 图注 |

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-publication-quality-plot.ipynb` | 拿 9.4 的交互效应数据, 用 `src/plotstyle.py` 从「默认 matplotlib」一步步升级到「出版级」(set_pub_style → 误差棒 → 去 chartjunk → 导出 PDF), before/after 对比; 再用 `schematic.py` 画一张 pipeline 方法图 |
| `notebooks/N2-figure-honesty.ipynb` | 用同一份数据画「截断 y 轴 (骗人) vs 诚实」两张图, 亲眼看 0.01 差距怎么被吹成翻倍, 并写一条 self-contained 图注 |

> 全程复用 9.4/9.5 的实验数据 (Robust-DPO 噪声鲁棒性) —— **一份数据走完「设计→留痕→出图」全生命周期**。

## 可复用模板 (`templates/`)

- `figure-checklist.md` — 出版级图投稿前 checklist (一图一信息/色盲/矢量/误差棒/诚实轴)
- `figure-caption.md` — self-contained 图注写作模板

## 工具 (`src/`)

- `plotstyle.py` — 出版级 matplotlib 样式包 (set_pub_style / Okabe-Ito 色盲色 / grouped_bar / save_figure 出 PDF+PNG)。**写真论文能直接用。**
- `schematic.py` — 用 matplotlib 画可复现的 pipeline 方法示意图

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。notebook 离线可跑 (复用 9.4 模拟器数据)。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 用一图一信息 / data-ink / 图类型 / 色彩 四维度评判任意图
- [ ] 用 `set_pub_style` 一行把默认图升级到出版级, 导出矢量 PDF
- [ ] 给任何均值图配正确的误差棒并在图注注明
- [ ] 设计一张 30 秒看懂的方法图 (Figure 1)
- [ ] 识别并避免截断 y 轴等 6 种误导, 写 self-contained 图注
- [ ] 让每张图对应一个 claim, 图注首句连读成论证骨架

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  执行   9.4 experiment-design ✅ / 9.5 experiment-ops-repro ✅
  输出   9.6 research-figures        ◄── 你在这里 (把结果画给人看)
        9.7 paper-writing-submission  (把研究写成论文)
        9.8 research-presentation     (把研究讲给人听)
  ...
```
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
