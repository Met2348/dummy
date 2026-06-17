"""
citation_graph.py — 用 Semantic Scholar 公开 API 拉一篇论文的引用关系, 画迷你图谱.

这是一个"真实科研动作"的小引擎, 也是本专题 (9.3) 通往 9.2 文献图谱的钩子:
- 读一篇论文时, 它的 **references** (它引了谁) 告诉你"它站在谁的肩膀上" → 补背景.
- 它的 **citations** (谁引了它) 告诉你"谁在它之上继续做" → 找最新进展、找 gap 是否已被填.

设计要点:
- 网络不可用 / API 限流时, 自动回退到内置样例数据 (围绕 DPO), 保证 notebook 永远能出图.
- 所有网络调用带超时, 失败返回 None, 绝不抛裸异常打断教学流程.

用法 (notebook):
    import citation_graph as cg
    data = cg.fetch_paper_graph("Direct Preference Optimization")  # 联网; 失败则用 sample
    G = cg.build_graph(data)
    cg.plot_graph(G)

用法 (命令行):
    python src/citation_graph.py "Direct Preference Optimization"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

# Windows 控制台默认 GBK; 强制 UTF-8 输出, 避免中文/特殊字符在命令行崩溃.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API = "https://api.semanticscholar.org/graph/v1"
TIMEOUT = 10


@dataclass
class PaperGraphData:
    center: str
    references: list[str] = field(default_factory=list)  # 它引用的 (更早)
    citations: list[str] = field(default_factory=list)   # 引用它的 (更新)
    source: str = "api"  # "api" 或 "sample"


# ----- 内置样例 (离线回退): 一个围绕 DPO 的极简引用邻域 -----
_SAMPLE = PaperGraphData(
    center="Direct Preference Optimization (DPO)",
    references=[
        "InstructGPT (RLHF, Ouyang 2022)",
        "PPO (Schulman 2017)",
        "Bradley-Terry preference model (1952)",
        "Learning to summarize from human feedback (2020)",
    ],
    citations=[
        "KTO: prospect-theoretic alignment (2024)",
        "ORPO: monolithic odds-ratio (2024)",
        "SimPO: reference-free (2024)",
        "Robust DPO under noisy preferences (2024)",
        "Iterative/Online DPO (2024)",
    ],
    source="sample",
)


def _get(url: str, params: dict) -> dict | None:
    """带超时的 GET; 任何失败返回 None (不打断教学)."""
    try:
        import requests
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _search_id(title: str) -> str | None:
    data = _get(f"{API}/paper/search", {"query": title, "limit": 1, "fields": "title"})
    if not data or not data.get("data"):
        return None
    return data["data"][0].get("paperId")


def fetch_paper_graph(title: str, limit: int = 8) -> PaperGraphData:
    """联网拉取 title 这篇的 references + citations; 失败则回退到内置样例.

    返回的 PaperGraphData.source 标明数据来自 'api' 还是 'sample'.
    """
    pid = _search_id(title)
    if pid is None:
        return _SAMPLE

    fields = "title,references.title,citations.title"
    data = _get(f"{API}/paper/{pid}", {"fields": fields})
    if not data:
        return _SAMPLE

    def _titles(key: str) -> list[str]:
        items = data.get(key) or []
        out = [it.get("title") for it in items if it.get("title")]
        return out[:limit]

    return PaperGraphData(
        center=data.get("title", title),
        references=_titles("references"),
        citations=_titles("citations"),
        source="api",
    )


def build_graph(data: PaperGraphData):
    """把 PaperGraphData 变成 networkx 有向图: references → center → citations."""
    import networkx as nx
    G = nx.DiGraph()
    c = data.center
    G.add_node(c, role="center")
    for r in data.references:
        G.add_node(r, role="reference")
        G.add_edge(r, c)   # 早 → 中心 (中心站在它肩上)
    for ci in data.citations:
        G.add_node(ci, role="citation")
        G.add_edge(c, ci)  # 中心 → 后续 (后续站在中心肩上)
    return G


def plot_graph(G, ax=None):
    """画图谱: references 在左, center 在中, citations 在右. 返回 matplotlib ax."""
    import matplotlib.pyplot as plt
    import networkx as nx

    roles = nx.get_node_attributes(G, "role")
    # 分层布局: 按角色定 x, 同层内均匀分布 y
    layer_x = {"reference": 0.0, "center": 1.0, "citation": 2.0}
    buckets: dict[str, list[str]] = {"reference": [], "center": [], "citation": []}
    for n, role in roles.items():
        buckets[role].append(n)
    pos = {}
    for role, nodes in buckets.items():
        for i, n in enumerate(sorted(nodes)):
            y = 0.0 if len(nodes) == 1 else 1.0 - 2.0 * i / (len(nodes) - 1)
            pos[n] = (layer_x[role], y)

    color = {"reference": "#9ecae1", "center": "#fc9272", "citation": "#a1d99b"}
    node_colors = [color[roles[n]] for n in G.nodes()]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.4,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title("引用邻域: 左=它引的(背景)  中=本文  右=引它的(后续)")
    ax.axis("off")
    return ax


def summarize(data: PaperGraphData) -> str:
    src = "联网获取" if data.source == "api" else "离线样例(网络不可用, 用内置 DPO 邻域演示)"
    lines = [f"中心论文: {data.center}   [{src}]",
             f"  ← 它引用了 {len(data.references)} 篇 (背景/肩膀):"]
    lines += [f"      · {r}" for r in data.references]
    lines += [f"  → 有 {len(data.citations)} 篇引用它 (后续/最新进展):"]
    lines += [f"      · {c}" for c in data.citations]
    lines.append("\n找 gap 提示: 看'引它的'那一列, 哪些角度已被做过? 哪些还没人碰?")
    return "\n".join(lines)


def main() -> int:
    title = sys.argv[1] if len(sys.argv) > 1 else "Direct Preference Optimization"
    data = fetch_paper_graph(title)
    print(summarize(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
