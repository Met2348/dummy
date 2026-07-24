"""
生成"poster版面留白/分区"好坏对照图:12格塞满到边缘的"文字墙" vs Better Poster风格的
大留白+单一论点+两侧细节栏。图的物理尺寸真实设成一张常见的36x48英寸(portrait)海报尺寸,
所以这里用的matplotlib fontsize(单位是pt=1/72英寸)和真实印出来海报上的实际字号是一一对应的、
不是随便取的相对值——可以直接和调研到的"字号-可读距离"真实数据表对照。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import textwrap

OUT = Path("_assets")
OUT.mkdir(exist_ok=True)

# 常见美国会议portrait poster物理尺寸(36x48英寸是常见默认值之一,并非所有会议通用——
# 具体尺寸每年、每个会议CFP都可能不同,这里只是为了让fontsize单位真实对应物理大小)
POSTER_W_IN, POSTER_H_IN = 36, 48
DPI = 80

FILLER = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
          "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam.")


def add_block(ax, x, y, w, h, label, fontsize, content_areas, n_lines=3, text_fontsize=None):
    """画一个内容矩形块(有边框),并把它的真实面积记进content_areas用于之后精确求和。
    正文按这个块的真实物理宽度(块宽度fraction x海报总宽英寸x72pt/英寸)手动textwrap,
    不依赖matplotlib的wrap=True(它是按整张画布宽度换行,不是按这个小方块的宽度换行,
    实测发现直接用wrap=True会导致窄栏里的文字越界写进相邻方块,这里改成显式按真实
    物理宽度换行,顺带演示"工具默认行为和你以为的不一样"这类真实的排版事故)。"""
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=1.0))
    content_areas.append(w * h)
    tfs = text_fontsize if text_fontsize is not None else fontsize
    ax.text(x + 0.004, y + h - 0.010, label, fontsize=fontsize, weight="bold", va="top")

    block_width_pt = w * POSTER_W_IN * 72
    avg_char_width_pt = tfs * 0.52  # DejaVu Sans经验值,正文用等宽近似估计换行宽度
    chars_per_line = max(int(block_width_pt / avg_char_width_pt) - 1, 8)
    long_filler = " ".join([FILLER] * (n_lines + 1))  # 保证素材足够长,能真正填满n_lines行
    wrapped = textwrap.fill(long_filler, width=chars_per_line)
    wrapped = "\n".join(wrapped.split("\n")[:n_lines])
    ax.text(x + 0.004, y + h - 0.028, wrapped, fontsize=tfs, va="top")


def measure_overflow(fig, ax, artists):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    overflow = []
    for a in artists:
        bbox = a.get_window_extent(renderer=renderer)
        if bbox.x0 < -2 or bbox.x1 > fig_w_px + 2 or bbox.y0 < -2 or bbox.y1 > fig_h_px + 2:
            overflow.append(getattr(a, "get_text", lambda: str(type(a)))()[:30] if hasattr(a, "get_text") else str(a))
    return overflow


def make_bad_poster():
    fig = plt.figure(figsize=(POSTER_W_IN, POSTER_H_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    content_areas = []

    # 标题栏:占满整行,字号不算小,但只是干巴巴的技术标题,不是"一句话讲结论"
    ax.add_patch(Rectangle((0.0, 0.955), 1.0, 0.045, fill=False, edgecolor="black", linewidth=1.0))
    content_areas.append(1.0 * 0.045)
    title_text = ax.text(0.5, 0.978, "Adaptive Test-Time Imagination Budget Allocation for World Models: A Study",
                          fontsize=30, ha="center", va="center", weight="bold")

    # 4列x3行,几乎无间隙地铺满剩余区域——典型"墙式海报"
    labels = ["1. Abstract", "2. Introduction", "3. Related Work",
              "4. Method Overview", "5. Controller Design", "6. Training Setup",
              "7. Baselines", "8. Main Results", "9. Ablations",
              "10. Discussion", "11. Limitations", "12. References"]
    n_cols, n_rows = 4, 3
    gap = 0.004  # 几乎没有留白
    grid_top, grid_bottom = 0.950, 0.02
    cell_w = (1.0 - (n_cols + 1) * gap) / n_cols
    cell_h = (grid_top - grid_bottom - (n_rows + 1) * gap) / n_rows
    body_fontsize = 11  # 低于调研到的24pt"最低可读字号"这条真实建议,故意演示反例
    for i, label in enumerate(labels):
        r, c = divmod(i, n_cols)
        x = gap + c * (cell_w + gap)
        y = grid_top - gap - (r + 1) * cell_h - r * gap
        add_block(ax, x, y, cell_w, cell_h, label, fontsize=13, content_areas=content_areas,
                  n_lines=6, text_fontsize=body_fontsize)

    all_texts = [t for t in ax.texts]
    overflow = measure_overflow(fig, ax, all_texts + [p for p in ax.patches])
    content_fraction = sum(content_areas)  # 各矩形按坐标定义时互不重叠,可以直接相加
    path = OUT / "bad_poster.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, {
        "content_fraction": content_fraction,
        "whitespace_fraction": 1 - content_fraction,
        "title_fontsize": title_text.get_fontsize(),
        "body_fontsize": body_fontsize,
        "n_blocks": len(labels),
        "overflow": overflow,
    }


def make_good_poster():
    fig = plt.figure(figsize=(POSTER_W_IN, POSTER_H_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    content_areas = []

    # 一整句话的大标题,直接讲结论,不是干巴巴的技术标题——Better Poster的核心主张
    title_w, title_h = 0.86, 0.10
    title_x, title_y = 0.07, 0.86
    ax.add_patch(Rectangle((title_x, title_y), title_w, title_h, fill=False, edgecolor="none"))
    content_areas.append(title_w * title_h)
    title_text = ax.text(0.5, title_y + title_h / 2,
                          "Imagination only pays off when it carries\ntask-relevant information the baseline lacks.",
                          fontsize=64, ha="center", va="center", weight="bold")

    # 中央大留白区域里放唯一的核心结论(大字号数字),四周留出真实的大量空白
    center_w, center_h = 0.42, 0.30
    center_x, center_y = 0.29, 0.42
    ax.add_patch(Rectangle((center_x, center_y), center_w, center_h, fill=False, edgecolor="#1a6fb0", linewidth=2.0))
    content_areas.append(center_w * center_h)
    ax.text(0.5, center_y + center_h * 0.62, "82.0%", fontsize=110, ha="center", color="#1a6fb0", weight="bold")
    ax.text(0.5, center_y + center_h * 0.22, "task-conditioned imagination hit rate (vs. 63.7% unconditioned)",
            fontsize=24, ha="center")

    # 左右两条细节栏,字号仍然守住24pt这条真实的最低可读建议,不是继续做小
    body_fontsize = 24
    left_labels = ["Method", "Related Work"]
    right_labels = ["Results Detail", "Contact / Code"]
    col_w = 0.16
    for i, label in enumerate(left_labels):
        y = 0.62 - i * 0.30
        add_block(ax, 0.03, y, col_w, 0.24, label, fontsize=22, content_areas=content_areas,
                  n_lines=5, text_fontsize=body_fontsize)
    for i, label in enumerate(right_labels):
        y = 0.62 - i * 0.30
        add_block(ax, 0.81, y, col_w, 0.24, label, fontsize=22, content_areas=content_areas,
                  n_lines=5, text_fontsize=body_fontsize)

    all_texts = [t for t in ax.texts]
    overflow = measure_overflow(fig, ax, all_texts + [p for p in ax.patches])
    content_fraction = sum(content_areas)
    path = OUT / "good_poster.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, {
        "content_fraction": content_fraction,
        "whitespace_fraction": 1 - content_fraction,
        "title_fontsize": title_text.get_fontsize(),
        "body_fontsize": body_fontsize,
        "n_blocks": 1 + len(left_labels) + len(right_labels),
        "overflow": overflow,
    }


if __name__ == "__main__":
    bad_path, bad_m = make_bad_poster()
    good_path, good_m = make_good_poster()

    print("bad_poster.png :", bad_m, "exists=", bad_path.exists())
    print("good_poster.png:", good_m, "exists=", good_path.exists())

    assert bad_m["overflow"] == [], f"坏例子不应该意外溢出画布: {bad_m['overflow']}"
    assert good_m["overflow"] == [], f"好例子的大字号也必须真的排得下: {good_m['overflow']}"

    assert bad_m["whitespace_fraction"] < 0.08, "坏例子(墙式排版)留白应该接近0"
    assert good_m["whitespace_fraction"] > 0.35, "好例子(Better Poster风格)留白应该超过调研到的'大量负空间'门槛"
    assert good_m["whitespace_fraction"] > bad_m["whitespace_fraction"] * 5, "好坏两例的留白比例差距应该非常悬殊"

    # 这两条直接对应调研到的真实字号规则:正文至少24pt、标题应远大于正文
    assert bad_m["body_fontsize"] < 24, "坏例子故意演示违反'正文最低24pt'这条真实调研到的建议"
    assert good_m["body_fontsize"] >= 24, "好例子的正文字号必须真的守住24pt这条下限"
    assert good_m["title_fontsize"] >= 60, "好例子标题字号应落在'12英尺外可读'量级(约60pt+)"

    print("ALL POSTER LAYOUT ASSERTIONS PASSED")
