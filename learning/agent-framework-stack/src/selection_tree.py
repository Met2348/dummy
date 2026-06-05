"""Framework selection decision tree."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SelectionCriteria:
    multi_agent: bool = False
    rag_heavy: bool = False
    typed_required: bool = False
    streaming_ui: bool = False
    vendor_lock: str = "any"   # "anthropic" | "openai" | "azure" | "any"
    language: str = "python"   # "python" | "typescript" | "csharp" | "java"
    enterprise: bool = False
    team_size: str = "small"   # "solo" | "small" | "enterprise"
    use_case: str = "general"  # "chatbot" | "research" | "rag" | "general"


def select_framework(c: SelectionCriteria) -> tuple[str, str]:
    if c.enterprise and c.language == "csharp":
        return "Semantic Kernel", "Enterprise C# stack."
    if c.vendor_lock == "anthropic":
        return "Claude Agent SDK", "Anthropic vendor + built-in tools."
    if c.vendor_lock == "openai":
        return "OpenAI Agents SDK", "Official OpenAI stack."
    if c.language == "typescript":
        if c.streaming_ui:
            return "Vercel AI SDK", "Edge + streaming UI."
        return "Vercel AI SDK", "TypeScript first."
    if c.multi_agent:
        return "LangGraph", "StateGraph + checkpoint + HITL."
    if c.rag_heavy:
        if c.enterprise:
            return "Haystack", "Enterprise search RAG."
        return "LlamaIndex", "5 index types + 3-line RAG."
    if c.typed_required:
        return "Pydantic AI", "Type-safe with Pydantic."
    if c.team_size == "solo" and c.use_case == "chatbot":
        return "CrewAI", "Quick role-based crew."
    return "LangChain", "General-purpose, biggest ecosystem."


def explain_path(c: SelectionCriteria) -> list[str]:
    notes = []
    if c.enterprise:
        notes.append(f"enterprise={c.enterprise}, language={c.language}")
    notes.append(f"vendor_lock={c.vendor_lock}")
    notes.append(f"language={c.language}")
    notes.append(f"multi_agent={c.multi_agent}, rag_heavy={c.rag_heavy}, typed={c.typed_required}")
    notes.append(f"team={c.team_size}, use_case={c.use_case}")
    framework, reason = select_framework(c)
    notes.append(f"-> {framework} (because: {reason})")
    return notes


def _self_test() -> None:
    c1 = SelectionCriteria(multi_agent=True)
    fw, _ = select_framework(c1)
    assert fw == "LangGraph", fw

    c2 = SelectionCriteria(rag_heavy=True)
    fw, _ = select_framework(c2)
    assert fw == "LlamaIndex", fw

    c3 = SelectionCriteria(vendor_lock="anthropic", multi_agent=True)
    fw, _ = select_framework(c3)
    assert fw == "Claude Agent SDK", "vendor_lock wins"

    c4 = SelectionCriteria(language="typescript", streaming_ui=True)
    fw, _ = select_framework(c4)
    assert fw == "Vercel AI SDK"

    c5 = SelectionCriteria(typed_required=True)
    fw, _ = select_framework(c5)
    assert fw == "Pydantic AI"

    c6 = SelectionCriteria(enterprise=True, language="csharp")
    fw, _ = select_framework(c6)
    assert fw == "Semantic Kernel"

    c7 = SelectionCriteria(team_size="solo", use_case="chatbot")
    fw, _ = select_framework(c7)
    assert fw == "CrewAI"

    c8 = SelectionCriteria()
    fw, _ = select_framework(c8)
    assert fw == "LangChain"

    path = explain_path(c1)
    assert any("LangGraph" in p for p in path)
    print("[OK] selection_tree._self_test passed (8 scenarios)")


if __name__ == "__main__":
    _self_test()
