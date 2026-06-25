"""研究 agent 的内核：ReAct 式 plan→search→draft→critique→revise 闭环。

四个零件在这里合体（对应 M7 的四个模块）：
- **规划 Planning**：decompose 把问题拆成子问题（再喂检索）。
- **工具 Tool use**：corpus.search 是外部检索工具。
- **记忆 Memory**：Scratchpad 累积每步观察（ReAct 的 scratchpad）。
- **多智能体 Roles**：生成由 MockLLM（Researcher），批判由 critic（Reviewer），职责分离。

`llm` 可插拔：默认 MockLLM（确定性模板，便于教学/测试）；接真 LLM 只需实现
decompose/draft_idea/revise 三个方法（duck typing）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import critic as critic_mod
from .corpus import search


@dataclass(frozen=True)
class Draft:
    title: str
    cited_ids: tuple
    claimed_novelty: float
    method: str
    has_baseline: bool


class Scratchpad:
    """ReAct 记忆：按步累积 (step, content)，可整体回看。"""

    def __init__(self):
        self.entries = []

    def add(self, step: str, content) -> None:
        self.entries.append((step, content))

    def render(self) -> str:
        out = []
        for i, (step, content) in enumerate(self.entries, 1):
            out.append(f"  [{i}] {step}: {content}")
        return "\n".join(out)


class MockLLM:
    """确定性的"语言模型"替身。draft_idea 会**故意混入一个检索集外的引用**，
    用来复现真实 LLM 的"幻觉引用"失败模式（好让 Reviewer 抓它）。"""

    HALLUCINATED_ID = "2599.00001"   # 一个看着像 arXiv id、但库里根本没有的引用

    def decompose(self, question: str) -> list:
        return [f"{question} method experiment",
                f"{question} evaluation pitfalls"]

    def draft_idea(self, question: str, retrieved) -> Draft:
        ids = tuple(p.arxiv_id for p in retrieved)
        method = ("结合检索到的 " + ", ".join(ids) +
                  " 的做法，提出一个端到端研究 agent 改进。")
        # ★ 关键：在真实引用后面追加一个幻觉引用——这是 LLM 常见错误的确定性复现
        cited = ids + (self.HALLUCINATED_ID,)
        return Draft(title=f"研究 idea：{question[:48]}",
                     cited_ids=cited, claimed_novelty=0.85,
                     method=method, has_baseline=False)

    def revise(self, draft: Draft, critique, retrieved) -> dict:
        retrieved_ids = {p.arxiv_id for p in retrieved}
        if critique is None:
            # 无 critic：原样采纳——幻觉引用、缺 baseline、虚高新颖度都留着
            return {
                "idea": draft.title,
                "citations": tuple(draft.cited_ids),
                "baseline": None,
                "novelty": draft.claimed_novelty,
                "limitations": (),
                "steps": ("（未经批判，直接产出）",),
            }
        # 有 critic：按 critique 修正
        grounded = tuple(c for c in draft.cited_ids if c in retrieved_ids)
        novelty = 0.5 if any("新颖度" in f for f in critique.flags) else draft.claimed_novelty
        baseline = "线性/小 MLP 基线" if any("baseline" in f for f in critique.flags) else None
        return {
            "idea": draft.title,
            "citations": grounded,                       # 幻觉引用已删
            "baseline": baseline,
            "novelty": novelty,
            "limitations": tuple(f"已修正：{s}" for s in critique.suggestions),
            "steps": ("拆解问题", "检索接地", "起草", "独立评审", "按评审修正"),
        }


class ResearchAgent:
    def __init__(self, llm=None, k_per_subq: int = 2):
        self.llm = llm or MockLLM()
        self.k = k_per_subq

    def run(self, question: str, use_critic: bool = True) -> dict:
        mem = Scratchpad()
        mem.add("plan", "拆解研究问题")
        subqs = self.llm.decompose(question)
        mem.add("subquestions", subqs)

        retrieved, seen = [], set()
        for sq in subqs:
            hits = search(sq, k=self.k)
            mem.add("search", f"{sq!r} -> {[h.arxiv_id for h in hits]}")
            for h in hits:
                if h.arxiv_id not in seen:
                    seen.add(h.arxiv_id)
                    retrieved.append(h)

        draft = self.llm.draft_idea(question, retrieved)
        mem.add("draft", f"引用 {list(draft.cited_ids)}，新颖度自评 {draft.claimed_novelty}")

        critique = critic_mod.review(draft, retrieved) if use_critic else None
        if critique is not None:
            mem.add("review", f"flags={list(critique.flags)}")

        plan = self.llm.revise(draft, critique, retrieved)
        mem.add("revise", f"最终引用 {list(plan['citations'])}")

        return {
            "question": question,
            "subquestions": subqs,
            "retrieved_ids": [p.arxiv_id for p in retrieved],
            "draft": draft,
            "critique": critique,
            "plan": plan,
            "transcript": mem.render(),
        }


def ungrounded_in(plan: dict, retrieved_ids) -> list:
    """最终计划里有几处引用不在检索集内（= 残留的幻觉引用）。"""
    rid = set(retrieved_ids)
    return [c for c in plan["citations"] if c not in rid]
