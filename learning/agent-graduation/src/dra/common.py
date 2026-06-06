"""Common types for DRA."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Plan:
    sub_questions: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DocSnippet:
    doc_id: str
    text: str
    url: str = ""


@dataclass
class Note:
    sub_question: str
    findings: list[DocSnippet] = field(default_factory=list)


@dataclass
class Citation:
    id: int
    text: str
    doc_id: str


@dataclass
class Draft:
    markdown: str
    claims: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Verified:
    final_markdown: str
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class DRACost:
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def usd(self, in_price: float = 3.0, out_price: float = 15.0,
            tool_price: float = 0.005) -> float:
        return round(
            self.tokens_in / 1e6 * in_price
            + self.tokens_out / 1e6 * out_price
            + self.tool_calls * tool_price,
            6,
        )


def _self_test() -> None:
    plan = Plan(sub_questions=["q1", "q2"], rationale="because")
    assert len(plan.sub_questions) == 2

    note = Note(sub_question="q1", findings=[DocSnippet("d1", "text")])
    assert note.findings[0].doc_id == "d1"

    draft = Draft(markdown="# Hi", claims=["c1"], citations=[Citation(id=1, text="src", doc_id="d1")])
    assert draft.claims == ["c1"]

    cost = DRACost(llm_calls=3, tool_calls=10, tokens_in=10000, tokens_out=2000)
    assert cost.usd() > 0
    print("[OK] dra.common._self_test passed")


if __name__ == "__main__":
    _self_test()
