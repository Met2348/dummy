# papers/ — research-figures 参考源

本专题教**可视化手艺**, papers/ 收信息可视化的经典与配色/工具资源。

## 信息可视化经典
- **Edward Tufte, *The Visual Display of Quantitative Information*** — data-ink ratio / chartjunk 的来源 (L1)。
- **Tufte, *Visual Explanations*** — 误导性图形的经典案例 (L4)。
- **Claus Wilke, *Fundamentals of Data Visualization*** (免费在线) — 现代、实用、含大量"好图 vs 坏图"对照, 强烈推荐。
- **Cleveland & McGill** — 人眼对各种视觉编码 (长度/角度/面积) 的感知精度研究 (为什么柱>饼, L1)。

## 配色 / 色盲
- **Okabe & Ito** 色盲安全调色板 (本专题 `plotstyle.OKABE_ITO`)。
- **viridis / cividis** 感知均匀色图 (Smith & van der Walt; 为什么别用 jet, L1/L4)。
- ColorBrewer (colorbrewer2.org) — 配色方案在线选择器。

## 工具文档
- matplotlib: https://matplotlib.org/  (尤其看 "Tight layout" / "Constrained layout" / rcParams)
- draw.io (diagrams.net): https://www.drawio.com/  (方法图, L3)
- Excalidraw: https://excalidraw.com/  · TikZ/PGF (LaTeX 矢量图)

## 为什么 papers/ 这么轻
本专题知识在**可跑的 `plotstyle.py` / `schematic.py`** 和课件审美里。最好的练习: 拿你某个复现专题的图,
用本专题 checklist 重做一遍, 对比 before/after。Wilke 的书是绝佳的"好图字典", 配合本专题用。
