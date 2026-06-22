"""
arxiv_triage.py — 把一批论文按你的研究关键词打分排序, 产出「本周阅读清单」.

为什么需要它: arxiv 每天上百篇, 全读不可能, 不读怕错过 —— 新手最常见的失败是
**焦虑性囤积** (开几十个标签页, 一篇没读)。解药是把「读 vs 不读」从情绪决策变成
一条可重复的打分流程: 给定你关心的关键词及权重, 自动算出每篇的相关分, 高分进精读队列,
低分扫一眼标题归档。这样你每周只花 30 分钟分诊, 而不是天天被信息流绑架。

打分模型 (故意简单、可解释):
    score(paper) = Σ_k  weight[k] × occurrences(k in title+abstract)
                   + TITLE_BONUS × (k 命中标题)
标题命中比摘要命中更值钱 (标题是作者认为最重要的词), 所以给额外加权。

纯 stdlib, 离线可跑 (内置 SAMPLE_FEED)。真要连真实 arxiv, 见 fetch_live() (可选, 需联网)。

用法 (notebook / import):
    from arxiv_triage import triage, SAMPLE_FEED
    ranked = triage(SAMPLE_FEED, {"reasoning": 3, "rl": 2, "alignment": 2})
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TITLE_BONUS = 2.0  # 关键词命中标题的额外加权倍率


def score_paper(paper: dict, weights: dict[str, float]) -> tuple[float, list[str]]:
    """返回 (分数, 命中的关键词列表). paper 需含 'title' 和 'abstract'."""
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    score = 0.0
    hits: list[str] = []
    for kw, w in weights.items():
        k = kw.lower()
        in_title = title.count(k)
        in_abs = abstract.count(k)
        if in_title or in_abs:
            hits.append(kw)
        score += w * in_abs
        score += w * TITLE_BONUS * in_title
    return score, hits


def triage(papers: list[dict], weights: dict[str, float]) -> list[dict]:
    """给每篇论文打分并按分数降序排序; 每篇附加 'score' 与 'hits' 字段."""
    ranked = []
    for p in papers:
        s, hits = score_paper(p, weights)
        ranked.append({**p, "score": round(s, 2), "hits": hits})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def reading_list(papers: list[dict], weights: dict[str, float], top_k: int = 5) -> list[dict]:
    """取分数 > 0 的前 top_k 篇作为本周精读清单 (分数为 0 的直接归档, 不进清单)."""
    ranked = [p for p in triage(papers, weights) if p["score"] > 0]
    return ranked[:top_k]


# ── 内置离线样例 feed (模拟某周 arxiv cs.CL 的一小撮论文) ─────────────────────
SAMPLE_FEED = [
    {
        "id": "2026.00001",
        "title": "Robust DPO under Noisy Preference Labels",
        "abstract": "We study direct preference optimization (DPO) when human preference "
        "annotations are noisy, and propose a robust alignment objective.",
    },
    {
        "id": "2026.00002",
        "title": "Process Reward Models for Multi-step Reasoning",
        "abstract": "A process reward model that scores each reasoning step improves "
        "math reasoning over outcome-only rewards.",
    },
    {
        "id": "2026.00003",
        "title": "A Survey of Vision Transformers for Medical Imaging",
        "abstract": "We survey ViT architectures applied to radiology and pathology images.",
    },
    {
        "id": "2026.00004",
        "title": "Scaling Laws for Reasoning RL",
        "abstract": "We measure how reinforcement learning for reasoning scales with "
        "model size, and find regime changes at the 30B boundary.",
    },
    {
        "id": "2026.00005",
        "title": "Efficient Quantization for Edge Deployment of CNNs",
        "abstract": "Post-training quantization scheme for convolutional networks on "
        "mobile hardware.",
    },
    {
        "id": "2026.00006",
        "title": "On the Reproducibility of RLHF Pipelines",
        "abstract": "We re-run several RLHF alignment pipelines and report large variance "
        "across seeds, questioning headline reasoning gains.",
    },
]


def fetch_live(query: str, max_results: int = 20) -> list[dict]:
    """(可选, 需联网) 用 arxiv 公开 API 拉一批论文. 失败则返回 SAMPLE_FEED 兜底,
    保证 notebook 在任何环境下都不报错。"""
    try:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        url = (
            "http://export.arxiv.org/api/query?"
            + urllib.parse.urlencode(
                {"search_query": query, "max_results": max_results}
            )
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            xml = resp.read().decode("utf-8")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(xml).findall("a:entry", ns):
            out.append(
                {
                    "id": (e.findtext("a:id", default="", namespaces=ns) or "").split("/")[-1],
                    "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()),
                }
            )
        return out or SAMPLE_FEED
    except Exception as exc:  # 离线 / 超时 / 限流 → 兜底, 不让 notebook 挂掉
        print(f"[arxiv_triage] live fetch 失败 ({exc!r}), 改用内置样例 feed.")
        return SAMPLE_FEED


if __name__ == "__main__":
    weights = {"reasoning": 3, "rl": 2, "dpo": 2, "alignment": 2, "reproducibility": 3}
    for i, p in enumerate(reading_list(SAMPLE_FEED, weights), 1):
        print(f"{i}. [{p['score']:>4}] {p['title']}  (命中: {', '.join(p['hits'])})")
