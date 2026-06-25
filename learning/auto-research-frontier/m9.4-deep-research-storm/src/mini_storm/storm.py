"""mini-STORM：多视角提问 → 检索 → 带 inline 引用的合成。

仿 STORM（[2402.14207](https://arxiv.org/abs/2402.14207)）的 outline-driven RAG：
先从多个视角提问、各自检索，再把检索到的事实合成带引用的段落。
**故意注入两句"引了真论文、但那论文不支持此论断"的不忠实句**——复现真实综述 agent 的典型错误。
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import retrieve


@dataclass(frozen=True)
class Sentence:
    text: str
    claim_tokens: frozenset   # 这句话主张的事实
    cited_doc: str | None     # 它引的文献 id


@dataclass(frozen=True)
class Report:
    topic: str
    perspectives: tuple
    retrieved_ids: tuple
    sentences: tuple


def perspectives(topic: str) -> list:
    """多视角提问：同一主题从不同角度问，覆盖更全（STORM 的核心招式）。"""
    return [
        f"{topic} retrieval outline perspective",
        f"{topic} pitfalls hallucination integrity",
        f"{topic} autonomous agent tree-search experiment",
    ]


def synthesize(topic: str, k_per_perspective: int = 2) -> Report:
    persp = perspectives(topic)
    retrieved, seen = [], set()
    for p in persp:
        for d in retrieve(p, k=k_per_perspective):
            if d.doc_id not in seen:
                seen.add(d.doc_id)
                retrieved.append(d)

    sentences = []
    # —— 忠实合成：每篇检索到的文档，挑它**真支持**的论断写一句，引它本身 ——
    for d in retrieved:
        claims = frozenset(sorted(d.supports)[:2])
        text = f"{d.title}：{', '.join(sorted(claims))} [{d.doc_id}]"
        sentences.append(Sentence(text, claims, d.doc_id))

    # —— 注入不忠实句①：claim 是 co-scientist 的"湿实验验证"，却引到 v2 ——
    sentences.append(Sentence(
        "AI Scientist v2 的结果已经过湿实验验证 [2504.08066]",
        frozenset({"wet-lab-validated"}), "2504.08066"))
    # —— 注入不忠实句②：claim 是 OpenScholar 的"高引用准确率"，却引到 STORM ——
    sentences.append(Sentence(
        "STORM 实现了高引用准确率 [2402.14207]",
        frozenset({"citation-accuracy"}), "2402.14207"))

    return Report(topic=topic, perspectives=tuple(persp),
                  retrieved_ids=tuple(d.doc_id for d in retrieved),
                  sentences=tuple(sentences))
