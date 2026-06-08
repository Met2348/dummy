"""Toy taxonomy helpers for the autonomous-agent survey."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import FrameworkRun


@dataclass
class MemoryRecord:
    text: str
    recency: float
    importance: float


def _overlap(query: str, text: str) -> float:
    q_terms = set(query.lower().split())
    if not q_terms:
        return 0.0
    t_terms = set(text.lower().split())
    return len(q_terms & t_terms) / len(q_terms)


def memory_read_score(
    query: str,
    memory: MemoryRecord,
    recency_weight: float = 1.0,
    relevance_weight: float = 1.0,
    importance_weight: float = 1.0,
) -> float:
    """Survey Eq. 1 in toy form: recency + relevance + importance."""
    return (
        recency_weight * memory.recency
        + relevance_weight * _overlap(query, memory.text)
        + importance_weight * memory.importance
    )


def select_memory(query: str, memories: list[MemoryRecord], k: int = 1) -> list[MemoryRecord]:
    return sorted(memories, key=lambda m: memory_read_score(query, m), reverse=True)[:k]


@dataclass
class AgentDesign:
    profile_method: str = "handcrafted"
    memory_structure: str = "hybrid"
    memory_operations: set[str] = field(default_factory=lambda: {"read", "write", "reflect"})
    planning: str = "with_feedback"
    action_uses_tools: bool = True
    capability_acquisition: str = "mechanism_engineering"


def architecture_summary(design: AgentDesign) -> list[str]:
    notes = [
        f"profile={design.profile_method}",
        f"memory_structure={design.memory_structure}",
        f"memory_ops={','.join(sorted(design.memory_operations))}",
        f"planning={design.planning}",
        f"tools={design.action_uses_tools}",
        f"capability={design.capability_acquisition}",
    ]
    if design.memory_structure == "unified":
        notes.append("risk=context_window_limit")
    if design.planning == "without_feedback":
        notes.append("risk=non_executable_initial_plan")
    if not design.action_uses_tools:
        notes.append("risk=limited_action_space")
    return notes


def framework_hint(design: AgentDesign) -> FrameworkRun:
    if design.planning == "with_feedback":
        return FrameworkRun(
            framework="LangGraph",
            loc=20,
            output="Use state graph, checkpoint, interrupt, and feedback edges.",
            abstraction_level="planning with feedback",
        )
    if design.memory_structure == "hybrid":
        return FrameworkRun(
            framework="LlamaIndex",
            loc=15,
            output="Use retrieval indexes and memory-backed query engines.",
            abstraction_level="hybrid memory",
        )
    return FrameworkRun(
        framework="LangChain",
        loc=10,
        output="Use runnable chains for straightforward action production.",
        abstraction_level="general chain",
    )


def _self_test() -> None:
    memories = [
        MemoryRecord("Alice asked about ReAct and tool feedback", recency=0.3, importance=0.6),
        MemoryRecord("Bob asked about CSS", recency=0.9, importance=0.1),
        MemoryRecord("Alice prefers LangGraph for feedback loops", recency=0.5, importance=0.9),
    ]
    selected = select_memory("Alice feedback", memories, k=2)
    assert "Alice" in selected[0].text
    assert any("LangGraph" in m.text for m in selected)

    design = AgentDesign()
    notes = architecture_summary(design)
    assert "memory_structure=hybrid" in notes
    assert "planning=with_feedback" in notes
    hint = framework_hint(design)
    assert hint.framework == "LangGraph"
    print("[OK] survey_taxonomy._self_test passed")


if __name__ == "__main__":
    _self_test()
