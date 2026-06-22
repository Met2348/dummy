"""
schematic.py — 用 matplotlib 画**方法示意图** (boxes-and-arrows pipeline), 让它也可复现.

为什么有这个: 论文里两类图 —— ① 数据图 (柱/线/散点, 见 plotstyle.py); ② 示意图
(讲方法的流程/架构, 论文的 Figure 1 通常是它)。示意图大多用 draw.io / Excalidraw / TikZ
手画, 但那样不可复现、改一次重画一次。本工具演示: 简单的 pipeline 示意图也能**代码化**,
和数据图一样进版本控制、一键重生成。复杂架构图仍建议专业工具, 但 pipeline/flow 用这个够。

核心:
  - draw_pipeline(stages, ...) : 横向 boxes + 箭头, 一行一个 pipeline
  - 配 plotstyle 的色盲安全色

纯 matplotlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def draw_pipeline(ax, stages, title="", box_color="#56B4E9", text_color="black"):
    """画一条横向 pipeline 示意图: stages 是 box 文字列表, 自动加箭头连接.

    stages: ["读论文", "找gap", "设计实验", "跑+留痕", "出图", "写论文"]
    """
    import matplotlib.patches as mpatches

    n = len(stages)
    box_w, box_h, gap = 1.0, 0.6, 0.5
    for i, label in enumerate(stages):
        x = i * (box_w + gap)
        box = mpatches.FancyBboxPatch(
            (x, 0), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor="#333333", facecolor=box_color)
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_h / 2, label, ha="center", va="center",
                fontsize=9, color=text_color, wrap=True)
        if i < n - 1:  # 箭头到下一个
            ax.annotate("", xy=(x + box_w + gap, box_h / 2),
                        xytext=(x + box_w, box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))
    ax.set_xlim(-0.3, n * (box_w + gap))
    ax.set_ylim(-0.3, box_h + 0.4)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams["axes.unicode_minus"] = False
    for f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:  # 中文字体, 避免 CJK 缺字警告
        try:
            mpl.rcParams["font.sans-serif"] = [f]; break
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(11, 1.8))
    draw_pipeline(ax, ["读论文\n找gap", "可证伪\n假设", "最小\n验证", "消融+\n方差", "出版级\n图", "写成\n论文"],
                  title="Module 9 研究流水线 (示意图也能代码化)")
    fig.savefig("_smoke_schematic.png", dpi=150, bbox_inches="tight")
    print("示意图已存: _smoke_schematic.png")
