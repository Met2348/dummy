"""
plotstyle.py — 出版级 matplotlib 样式包: 一行把"能看"的默认图升级成"能印"的论文图.

为什么需要它: 顶会论文的图有一套不成文但一致的规范 (字号够大、矢量格式、色盲安全、
去掉多余墨水)。每次手调 rcParams 既慢又不一致。这个工具把规范固化成一个函数, 你以后写
真论文直接 import 用 —— 这是本专题最实用的产出, 不是玩具。

核心:
  - set_pub_style()    : 一行设好出版级 rcParams (字号/线宽/字体/留白)
  - OKABE_ITO          : 色盲安全调色板 (Okabe-Ito, 8 色, 学术界推荐)
  - grouped_bar(...)   : 带误差棒的分组柱状图 (直接吃 9.4/9.5 的 mean/std)
  - save_figure(...)   : 同时存矢量 PDF (投稿用) + PNG (预览用), 统一 DPI/bbox

纯 matplotlib (+ numpy)。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Okabe-Ito 色盲安全调色板 (8 色). 约 8% 男性有色觉障碍, 用它你的图对所有人可读。
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


def set_pub_style(base_fontsize: int = 11):
    """一行设好出版级 rcParams. 在画图前调用一次.

    设计依据 (每条都对应"为什么默认图不能印"):
      - 字号调大: 论文图会被缩到单栏宽 (~3.3in), 默认字号缩完看不清。
      - 线宽/marker 调大: 同理, 缩小后要还看得见。
      - 去掉上/右边框: 减少 chartjunk (Tufte 的 data-ink 原则)。
      - 色盲安全色环 + 矢量字体: 可读性 + 可印刷。
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "axes.spines.top": False,      # 去掉上边框 (data-ink)
        "axes.spines.right": False,    # 去掉右边框
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),  # 色盲安全色环
        "figure.dpi": 110,
        "savefig.dpi": 300,            # 印刷级
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,   # 负号正常显示
    })
    # 中文字体 (有则用, 无则退到默认, 不报错)
    for f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            mpl.rcParams["font.sans-serif"] = [f]
            break
        except Exception:
            pass


def column_figsize(width_in: float = 3.3, aspect: float = 0.72):
    """返回适合论文单栏宽 (~3.3in) 的 (w, h). 双栏图用 width_in=6.9."""
    return (width_in, width_in * aspect)


def grouped_bar(ax, groups, series, means, errs=None, ylabel="", title="",
                colors=None):
    """带误差棒的分组柱状图. 直接吃 9.4/9.5 的聚合结果.

    groups: x 轴组名 (如 ['noise=0','noise=0.2','noise=0.4'])
    series: 每组里的系列名 (如 ['DPO','Robust-DPO'])
    means:  shape (len(series), len(groups)) 的均值
    errs:   同形状的误差 (如 SEM), 可 None
    """
    means = np.asarray(means, dtype=float)
    n_series, n_groups = means.shape
    x = np.arange(n_groups)
    w = 0.8 / n_series
    colors = colors or OKABE_ITO
    for i, name in enumerate(series):
        e = None if errs is None else np.asarray(errs)[i]
        ax.bar(x + i * w, means[i], w, yerr=e, capsize=4,
               label=name, color=colors[i % len(colors)],
               error_kw={"elinewidth": 1.0})
    ax.set_xticks(x + w * (n_series - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def save_figure(fig, stem, out_dir=".", formats=("pdf", "png")):
    """同时存矢量 PDF (投稿) + PNG (预览). 返回写出的路径列表.

    为什么 PDF: 矢量格式无限缩放不糊, 是投稿的标准 (位图 PNG 放大会马赛克)。
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    set_pub_style()
    fig, ax = plt.subplots(figsize=column_figsize())
    grouped_bar(
        ax,
        groups=["0.0", "0.2", "0.4"],
        series=["DPO", "Robust-DPO"],
        means=[[0.62, 0.52, 0.40], [0.62, 0.60, 0.53]],
        errs=[[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],
        ylabel="win-rate", title="噪声鲁棒性",
    )
    ax.set_xlabel("偏好标签噪声")
    paths = save_figure(fig, "demo", out_dir="_smoke")
    print("出版级图已存:", [str(p) for p in paths])
