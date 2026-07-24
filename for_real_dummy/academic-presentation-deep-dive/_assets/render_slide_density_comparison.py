"""
生成"信息密度差"对比图:一页塞8个要点 vs 一页1个要点。
真实用 matplotlib 渲染两张 16:9 幻灯片版式,并用多个真实测量到的指标做量化对比——
不是只挑一个显得好看的指标,几个指标算出来后如实报告,包括一个和直觉不完全一致的地方。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("_assets")
OUT.mkdir(exist_ok=True)

SLIDE_W, SLIDE_H = 13.33, 7.5  # 16:9,和真实投影幻灯片比例一致
DPI = 150


def measure(fig, texts):
    """对一页幻灯片的全部文字对象做真实测量,不是凭印象断言。
    额外做一项"是否溢出画布"的真实检查——文字溢出是幻灯片制作最常见、最尴尬的真实事故之一。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    ink_area = 0.0
    overflow = []
    for t in texts:
        bbox = t.get_window_extent(renderer=renderer)
        ink_area += max(bbox.width, 0) * max(bbox.height, 0)
        if bbox.x0 < -1 or bbox.x1 > fig_w_px + 1 or bbox.y0 < -1 or bbox.y1 > fig_h_px + 1:
            overflow.append(t.get_text()[:30])
    total_chars = sum(len(t.get_text()) for t in texts)
    total_words = sum(len(t.get_text().split()) for t in texts)
    fontsizes = [t.get_fontsize() for t in texts]
    return {
        "n_elements": len(texts),
        "ink_fraction": ink_area / (fig_w_px * fig_h_px),
        "total_chars": total_chars,
        "total_words": total_words,
        "min_fontsize": min(fontsizes),
        "max_fontsize": max(fontsizes),
        "overflow": overflow,
    }


def make_bad_slide():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    texts = []
    texts.append(fig.text(0.02, 0.95, "Experimental Results", fontsize=18, weight="bold"))

    bullets = [
        "Our method improves accuracy over the baseline on all four evaluated benchmark tasks",
        "We use a learning rate of 3e-4 with cosine decay and 500 warmup steps for stability",
        "The controller adds only 1.2% parameter overhead relative to the backbone world model",
        "Ablation shows removing the gating head drops performance by 14.7 percentage points",
        "We evaluate on 5 random seeds and report mean plus standard deviation for every table",
        "Baseline comparisons include AVIC, FFDC and Video-T1 under matched compute budgets",
        "Wall-clock latency is reduced by 30 percent compared to the always-on imagination policy",
        "Future work includes scaling to real robot manipulation tasks beyond the simulator",
    ]
    y0, dy = 0.88, 0.088
    for i, b in enumerate(bullets):
        texts.append(fig.text(0.03, y0 - i * dy, "• " + b, fontsize=13, wrap=True))

    # 底部再塞一个密密麻麻的小表格,进一步挤占空间、字号进一步缩小
    table_text = "H=1: 0.10  H=2: 0.13  H=3: 0.16  H=5: 0.17  H=8: 0.23  K=1: 0.36  K=3: 0.17  K=5: 0.18  K=10: 0.10"
    texts.append(fig.text(0.03, 0.06, table_text, fontsize=9, family="monospace"))

    m = measure(fig, texts)
    path = OUT / "bad_slide.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, m


def make_good_slide():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    texts = []
    # 一句话讲完整个take-home message,大字号——这是全页唯一的"论点"
    headline = "Imagination only helps if it knows\nsomething the baseline doesn't."
    texts.append(fig.text(0.08, 0.80, headline, fontsize=32, weight="bold", va="top"))

    # 一个支撑性的大数字,不是一整张表格
    texts.append(fig.text(0.08, 0.38, "82.0%", fontsize=64, color="#1a6fb0", weight="bold"))
    texts.append(fig.text(0.08, 0.28, "task-conditioned imagination hit rate\nvs. 63.7% unconditioned", fontsize=16))

    # 一行出处,字号明显更小,不抢主视觉——但仍然刻意保持在演讲厅可读的下限之上
    texts.append(fig.text(0.08, 0.06, "3 target objects, 5 seeds -- see paper Table 2 for full breakdown", fontsize=11, color="gray"))

    m = measure(fig, texts)
    path = OUT / "good_slide.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, m


if __name__ == "__main__":
    bad_path, bad_m = make_bad_slide()
    good_path, good_m = make_good_slide()

    print("bad_slide.png :", bad_m, "exists=", bad_path.exists())
    print("good_slide.png:", good_m, "exists=", good_path.exists())

    # ---- 真实断言,四个指标里三个符合"好例子信息更少更大"的直觉,一个不符合,如实全部保留 ----
    assert bad_m["n_elements"] > good_m["n_elements"], "坏例子的独立信息块数量应该更多"
    assert bad_m["total_words"] > good_m["total_words"] * 3, "坏例子的总字数应该远超好例子"
    assert bad_m["total_chars"] > good_m["total_chars"] * 3, "坏例子的总字符数应该远超好例子"
    assert bad_m["min_fontsize"] < good_m["min_fontsize"], "坏例子里最小的字号应该比好例子更小(更难从后排看清)"
    assert good_m["max_fontsize"] > bad_m["max_fontsize"] * 2, "好例子的核心论点字号应该远大于坏例子任何一处文字"
    assert bad_m["overflow"] == [], f"坏例子不应该有文字溢出画布(这里如果溢出,是另一种真实事故,不是本例要演示的): {bad_m['overflow']}"
    assert good_m["overflow"] == [], f"好例子的大字号标题必须真的排得下,不能'为了大而溢出画布': {good_m['overflow']}"

    # ink_fraction 这一项如实报告、不断言方向——见 02 号文件正文对这个反直觉结果的讨论
    print(f"[for-the-record] ink_fraction bad={bad_m['ink_fraction']:.4f} good={good_m['ink_fraction']:.4f} "
          f"(注意:这一项不是'坏例子更低'就代表更克制,原因见正文讨论)")
    print("ALL DENSITY ASSERTIONS PASSED")
