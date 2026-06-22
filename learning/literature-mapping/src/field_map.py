"""
field_map.py — 把一组论文整理成「领域地图」: 方法谱系 (taxonomy) + 演进时间线 (timeline).

为什么需要它 (9.2 的产出): 滚雪球 (snowball.py) 给你一张引用网, 但网还不是「地图」。
地图要回答三个问题, 这正是导师面试或开题时会问你的:
  1. 这个子领域有哪几个**流派 (方法族)**? —— taxonomy
  2. SOTA 是怎么一步步**演进**过来的? —— timeline
  3. 现在的**前线/争论**在哪? —— frontier (接 9.3 找 gap)

本工具把 snowball 的网 + 每篇的 (年份, 方法族) 标注, 渲染成:
  - 一张 markdown 领域地图表 (可直接进你的第二大脑, 9.1),
  - 一张 matplotlib 时间线/谱系图 (可进 mini-survey, 也是 9.6 出版级图的预演)。

纯 stdlib + matplotlib。配合 snowball.py 的内置数据集离线可跑。
"""
from __future__ import annotations

import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def papers_from_net(net) -> list[dict]:
    """从 snowball 的 networkx 图抽出论文列表 (含被引数), 给本模块用."""
    out = []
    for n in net.nodes():
        d = net.nodes[n]
        out.append({"id": n, "name": d["name"], "year": d["year"],
                    "family": d["family"], "cited_by": net.in_degree(n)})
    return out


def taxonomy(papers: list[dict]) -> dict[str, list[dict]]:
    """按方法族分组 (流派). 返回 {family: [papers...]}, 组内按年份排序."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        groups[p["family"]].append(p)
    for fam in groups:
        groups[fam].sort(key=lambda p: p["year"])
    return dict(groups)


def timeline(papers: list[dict]) -> dict[int, list[dict]]:
    """按年份分组 (演进线). 返回 {year: [papers...]}."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for p in papers:
        groups[p["year"]].append(p)
    return dict(sorted(groups.items()))


def to_markdown_map(papers: list[dict], title: str = "子领域领域地图") -> str:
    """渲染一张 markdown 领域地图: 按流派分块, 标出每篇年份与被引数, 奠基作加 ★."""
    tax = taxonomy(papers)
    maxcite = max((p["cited_by"] for p in papers), default=0)
    lines = [f"# {title}\n", f"> 共 {len(papers)} 篇, {len(tax)} 个流派. ★ = 该领域奠基作 (被引最多).\n"]
    for fam, ps in sorted(tax.items(), key=lambda kv: -max(p["cited_by"] for p in kv[1])):
        lines.append(f"## 流派: {fam}")
        for p in ps:
            star = " ★" if p["cited_by"] == maxcite and maxcite > 0 else ""
            lines.append(f"- **{p['name']}** ({p['year']}) — 被引 {p['cited_by']}{star}")
        lines.append("")
    lines.append("## 我的位置 (填)")
    lines.append("- 我的复现/方向落在哪个流派: ____")
    lines.append("- 我最该补读的奠基作: ____")
    lines.append("- 我看到的前线/争论 (→ 9.3 找 gap): ____")
    return "\n".join(lines)


def plot_timeline(papers: list[dict], ax=None):
    """画演进时间线: x=年份, y=方法族, 点大小=被引数. 一眼看出每个流派何时兴起."""
    import matplotlib.pyplot as plt

    fams = sorted({p["family"] for p in papers})
    fam_y = {f: i for i, f in enumerate(fams)}
    palette = plt.cm.tab10.colors
    fam_color = {f: palette[i % len(palette)] for i, f in enumerate(fams)}

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    for p in papers:
        ax.scatter(p["year"], fam_y[p["family"]],
                   s=120 + 90 * p["cited_by"], color=fam_color[p["family"]],
                   alpha=0.8, edgecolors="white", zorder=3)
        ax.annotate(p["name"].split(" (")[0], (p["year"], fam_y[p["family"]]),
                    fontsize=6.5, xytext=(0, 10), textcoords="offset points",
                    ha="center")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=9)
    ax.set_xlabel("年份")
    ax.set_title("子领域演进时间线 (点越大 = 被引越多)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


if __name__ == "__main__":
    import snowball as sb
    net = sb.load_sample_net()
    papers = papers_from_net(net)
    print(to_markdown_map(papers, "偏好优化子领域地图"))
