"""一个 mock 的"arXiv 检索库"——取 papers/ 库里的一小撮真论文，做确定性关键词检索。

这是研究 agent 的**工具层（tool）**：把"检索"这个外部能力抽象成一个可调用的 search()。
真实 agent 会接 arXiv API / 向量库；这里用关键词重叠，纯 CPU、确定性，便于教学与测试。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    keywords: tuple   # 用于检索的词
    gist: str


# 取自 papers/INDEX.md 的真实条目（节选）
CORPUS = (
    Paper("2408.06292", "The AI Scientist",
          ("end-to-end", "ideation", "experiment", "writeup", "review", "automated"),
          "全自动五阶段科研流水线（模板法）。"),
    Paper("2504.08066", "The AI Scientist v2",
          ("tree-search", "agentic", "experiment-manager", "vlm", "autonomous"),
          "去模板、树搜索 + 实验经理 agent + 看图反馈。"),
    Paper("2502.18864", "Multi-Agent Research Systems",
          ("multi-agent", "roles", "debate", "collaboration", "orchestration"),
          "多智能体角色分工做研究。"),
    Paper("2501.04227", "Agent Laboratory",
          ("roles", "phd", "postdoc", "pipeline", "experiment", "writeup"),
          "PhD/Postdoc/Professor 角色流水线把 idea 做成论文。"),
    Paper("2409.04109", "Can LLMs Generate Novel Research Ideas?",
          ("ideation", "novelty", "human-study", "ranking", "creativity"),
          "大规模人类对照：LLM 点子被评得更新颖。"),
    Paper("2506.20803", "The Ideation-Execution Gap",
          ("ideation", "execution", "feasibility", "gap", "evaluation"),
          "把 idea 真去执行：AI 点子表现反不如人类。"),
    Paper("2402.14207", "Assisting Writing with STORM",
          ("retrieval", "perspective", "outline", "writeup", "citation"),
          "多视角提问 + outline-driven RAG 写综述。"),
    Paper("2505.13259", "A Survey on LLM Agents for Science",
          ("survey", "taxonomy", "tool", "analyst", "scientist", "autonomy"),
          "Tool→Analyst→Scientist 自主性阶梯综述。"),
    Paper("2509.08713", "Hidden Pitfalls of AI Research Agents",
          ("pitfalls", "hallucination", "benchmark-gaming", "integrity", "critique"),
          "刷榜、幻觉结果、伪造数据集等隐蔽陷阱。"),
)

CORPUS_IDS = frozenset(p.arxiv_id for p in CORPUS)


def _tokens(text: str) -> set:
    return {t.strip(".,?!").lower() for t in text.replace("-", " ").split() if t}


def search(query: str, k: int = 2, corpus=CORPUS) -> list:
    """关键词重叠检索：按 query 与 (title+keywords) 的命中数排序，确定性、可复现。

    平手按 arxiv_id 排序，保证多次调用结果完全一致（确定性是可测试的前提）。
    """
    q = _tokens(query)
    scored = []
    for p in corpus:
        hay = _tokens(p.title) | set(p.keywords)
        score = len(q & hay)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda sp: (-sp[0], sp[1].arxiv_id))
    return [p for _, p in scored[:k]]
