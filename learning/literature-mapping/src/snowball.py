"""
snowball.py — 把一个子领域的论文 + 引用关系建成网络, 滚雪球扩展, 识别奠基作与前沿作.

为什么需要它 (9.2 的核心动作): 摸清一个子领域, 不是把 arxiv 上带关键词的论文都读一遍 ——
那样既漏 (新词没覆盖) 又淹 (太多)。正确做法是**滚雪球 (snowballing)**:
  - 后向 (backward): 一篇论文「引了谁」→ 它站在谁的肩膀上 → 顺藤摸到**奠基作 (seminal)**。
  - 前向 (forward): 「谁引了它」→ 谁在它之上继续做 → 顺藤摸到**前沿作 (frontier)**。
反复滚几轮, 一个子领域的骨架就浮出来了。

网络建好后, 用**中心度 (centrality)** 量化每篇论文的地位:
  - 入度高 (被很多篇引) = 奠基性, 是你必须先读的「肩膀」。
  - 出度高 / 年份新 = 前沿, 是你找 gap 的地方 (9.3)。

纯 networkx, 内置一个真实的「偏好优化 (preference optimization)」子领域离线数据集,
所以 notebook 不联网也能跑出完整领域图。

用法:
    import snowball as sb
    net = sb.load_sample_net()           # 离线样例: DPO/preference-optimization 子领域
    print(sb.seminal_papers(net, k=3))   # 奠基作
    print(sb.frontier_papers(net, k=3))  # 前沿作
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── 内置离线数据集: 偏好优化 (preference optimization) 子领域 ───────────────
# 每个 paper: id -> (短名, 年份, 方法族). edges: (引用者, 被引者) = "a 引了 b".
PAPERS: dict[str, tuple[str, int, str]] = {
    "bt1952":      ("Bradley-Terry 偏好模型", 1952, "理论基础"),
    "ppo2017":     ("PPO", 2017, "RL 基础"),
    "summ2020":    ("Learning to summarize from HF", 2020, "RLHF 早期"),
    "instructgpt": ("InstructGPT (RLHF)", 2022, "RLHF 经典"),
    "dpo2023":     ("DPO", 2023, "直接偏好优化"),
    "ipo2023":     ("IPO", 2023, "直接偏好优化"),
    "kto2024":     ("KTO", 2024, "直接偏好优化"),
    "orpo2024":    ("ORPO", 2024, "直接偏好优化"),
    "simpo2024":   ("SimPO (reference-free)", 2024, "直接偏好优化"),
    "rdpo2024":    ("Robust-DPO (噪声偏好)", 2024, "鲁棒性"),
    "online2024":  ("Iterative/Online DPO", 2024, "在线化"),
    "r1_2025":     ("DeepSeek-R1 (RL for reasoning)", 2025, "推理 RL"),
}

# "a 引了 b" 的边 (a 更晚, 站在 b 肩上)
EDGES: list[tuple[str, str]] = [
    ("summ2020", "ppo2017"), ("summ2020", "bt1952"),
    ("instructgpt", "ppo2017"), ("instructgpt", "summ2020"),
    ("dpo2023", "instructgpt"), ("dpo2023", "bt1952"), ("dpo2023", "ppo2017"),
    ("ipo2023", "dpo2023"), ("ipo2023", "bt1952"),
    ("kto2024", "dpo2023"), ("orpo2024", "dpo2023"),
    ("simpo2024", "dpo2023"), ("simpo2024", "ipo2023"),
    ("rdpo2024", "dpo2023"), ("rdpo2024", "ipo2023"),
    ("online2024", "dpo2023"), ("online2024", "instructgpt"),
    ("r1_2025", "ppo2017"), ("r1_2025", "instructgpt"),
]


def load_sample_net():
    """返回内置偏好优化子领域的 networkx 有向图 (边: 引用者 → 被引者)."""
    import networkx as nx
    G = nx.DiGraph()
    for pid, (name, year, fam) in PAPERS.items():
        G.add_node(pid, name=name, year=year, family=fam)
    G.add_edges_from(EDGES)
    return G


def seminal_papers(net, k: int = 3) -> list[dict]:
    """奠基作 = 被引最多 (入度最高). 这些是你摸领域时必须先读的「肩膀」."""
    ranked = sorted(net.nodes(), key=lambda n: net.in_degree(n), reverse=True)
    return [_info(net, n, cited_by=net.in_degree(n)) for n in ranked[:k]]


def frontier_papers(net, k: int = 3) -> list[dict]:
    """前沿作 = 年份最新且还没什么人引 (入度低). 这是找 gap 的前线."""
    leaves = [n for n in net.nodes()]
    ranked = sorted(leaves, key=lambda n: (net.nodes[n]["year"], -net.out_degree(n)), reverse=True)
    return [_info(net, n, cited_by=net.in_degree(n)) for n in ranked[:k]]


def centrality(net) -> list[dict]:
    """用 PageRank 量化每篇论文在子领域里的「重要性」, 降序返回."""
    import networkx as nx
    # 反向图上跑 PageRank: 被引越多越重要 (PageRank 默认顺着边走, 我们要"被指向"得分高)
    pr = nx.pagerank(net.reverse())
    ranked = sorted(net.nodes(), key=lambda n: pr[n], reverse=True)
    return [{**_info(net, n, cited_by=net.in_degree(n)), "pagerank": round(pr[n], 4)}
            for n in ranked]


def snowball(net, seeds: list[str], direction: str = "backward", rounds: int = 1) -> set[str]:
    """从 seeds 出发滚雪球. direction='backward' 收集它们引的; 'forward' 收集引它们的."""
    frontier = set(seeds)
    collected = set(seeds)
    for _ in range(rounds):
        nxt: set[str] = set()
        for n in frontier:
            neigh = net.successors(n) if direction == "backward" else net.predecessors(n)
            nxt.update(neigh)
        nxt -= collected
        collected |= nxt
        frontier = nxt
        if not frontier:
            break
    return collected


def _info(net, n: str, **extra) -> dict:
    d = net.nodes[n]
    return {"id": n, "name": d["name"], "year": d["year"], "family": d["family"], **extra}


def plot_net(net, ax=None):
    """按年份分层画引用网: x=年份, 节点大小=被引数, 颜色=方法族. 返回 ax."""
    import matplotlib.pyplot as plt
    import networkx as nx

    fams = sorted({net.nodes[n]["family"] for n in net.nodes()})
    palette = plt.cm.tab10.colors
    fam_color = {f: palette[i % len(palette)] for i, f in enumerate(fams)}

    # x = 年份, y = 同年内错开
    by_year: dict[int, list[str]] = {}
    for n in net.nodes():
        by_year.setdefault(net.nodes[n]["year"], []).append(n)
    pos = {}
    for year, nodes in by_year.items():
        for i, n in enumerate(sorted(nodes)):
            y = 0 if len(nodes) == 1 else (i - (len(nodes) - 1) / 2)
            pos[n] = (year, y)

    sizes = [300 + 350 * net.in_degree(n) for n in net.nodes()]
    colors = [fam_color[net.nodes[n]["family"]] for n in net.nodes()]
    labels = {n: net.nodes[n]["name"].split(" (")[0] for n in net.nodes()}

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 7))
    nx.draw_networkx_edges(net, pos, ax=ax, alpha=0.25, arrows=True,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(net, pos, ax=ax, node_size=sizes, node_color=colors)
    nx.draw_networkx_labels(net, pos, ax=ax, labels=labels, font_size=7)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=9, label=f) for f, c in fam_color.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="方法族")
    ax.set_xlabel("年份"); ax.set_title("子领域引用网 (节点越大=被引越多=越奠基)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    return ax


if __name__ == "__main__":
    net = load_sample_net()
    print("子领域:", net.number_of_nodes(), "篇,", net.number_of_edges(), "条引用\n")
    print("奠基作 (必读肩膀):")
    for p in seminal_papers(net, 3):
        print(f"  · {p['name']} ({p['year']})  被引 {p['cited_by']} 次")
    print("前沿作 (找 gap 前线):")
    for p in frontier_papers(net, 3):
        print(f"  · {p['name']} ({p['year']})")
