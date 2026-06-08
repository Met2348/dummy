"""Capstone - 3-layer memory chatbot: 10-turn recall test."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import reset_mock_time
from mem0_mock import Mem0Store, extract_facts
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from personalization import UserProfile


TURNS = [
    "Hi, my name is Alice. I prefer Anthropic Claude over GPT.",
    "What's the weather today?",
    "Tell me about RAG.",
    "What's RAG-Fusion?",
    "How does ColBERT work?",
    "What is GraphRAG?",
    "Pros and cons of HippoRAG?",
    "Compare BM25 to dense retrieval.",
    "Which embedding model is best?",
    "What was my preferred LLM?",
]


@dataclass
class ChatResult:
    turns: list[dict] = field(default_factory=list)
    profile: UserProfile | None = None
    episodic_count: int = 0
    final_recall: str = ""
    recall_source: str = ""


def _mock_answer(user_text: str, profile: UserProfile, episodic: EpisodicMemory, sem: SemanticMemory) -> tuple[str, str]:
    text = user_text.lower()
    if "preferred" in text and "llm" in text:
        pref = profile.preferences.get("llm")
        if pref:
            return f"Your preferred LLM is {pref.title()}.", "semantic_profile"
        results = sem.search("Alice prefers LLM", user_id="alice", k=3)
        if results:
            return f"From memory: {results[0].object}.", "semantic_search"
        return "I don't remember.", "miss"
    return f"(mock answer for: {user_text[:50]})", "default"


def run_capstone() -> ChatResult:
    reset_mock_time()
    profile = UserProfile(user_id="alice")
    mem0 = Mem0Store()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    result = ChatResult(profile=profile)

    for i, user_text in enumerate(TURNS, start=1):
        episodic.add("alice", "user", user_text)
        facts = extract_facts(user_text)
        for f in facts:
            kind = f["kind"]
            value = f["value"]
            if kind == "name":
                profile.name = value
                semantic.add_triple("Alice", "name", value, "alice")
            elif kind == "preference":
                value_t = value.title()
                old = profile.preferences.get("llm")
                if old and old.lower() != value_t.lower():
                    profile.set_preference("llm", value_t)
                    semantic.update_triple("Alice", "prefers", value_t, "alice")
                else:
                    profile.set_preference("llm", value_t)
                    semantic.add_triple("Alice", "prefers", value_t, "alice")
            elif kind == "role":
                profile.domain["role"] = value
        mem0.add(user_text, "alice")

        answer, source = _mock_answer(user_text, profile, episodic, semantic)
        episodic.add("alice", "agent", answer)
        result.turns.append({
            "turn": i, "user": user_text, "agent": answer, "source": source,
        })

    result.episodic_count = len([e for e in episodic.episodes if e.user_id == "alice"])
    result.final_recall = result.turns[-1]["agent"]
    result.recall_source = result.turns[-1]["source"]
    return result


def to_md(r: ChatResult) -> str:
    lines = [
        "# 3-Layer Memory Chatbot Capstone\n",
        "## Profile after 10 turns",
    ]
    p = r.profile
    lines.append(f"- name: {p.name}")
    for k, v in p.preferences.items():
        lines.append(f"- preference[{k}]: {v}")

    lines.append(f"\n## Episodic: {r.episodic_count} events stored")
    lines.append("\n## Turn-by-turn")
    lines.append("| # | User | Agent | Source |")
    lines.append("|---|------|-------|--------|")
    for t in r.turns:
        u = t["user"][:40]
        a = t["agent"][:40]
        lines.append(f"| {t['turn']} | {u} | {a} | {t['source']} |")

    pass_ = ("Anthropic" in r.final_recall) and (r.recall_source == "semantic_profile")
    verdict = "[PASS]" if pass_ else "[FAIL]"
    lines.append(f"\n## Verdict: {verdict}")
    lines.append(f"- Final recall: {r.final_recall}")
    lines.append(f"- Recall source: {r.recall_source}")
    return "\n".join(lines)


def _self_test() -> None:
    r = run_capstone()
    assert r.profile is not None
    assert r.profile.name == "Alice", r.profile.name
    assert "Anthropic" in r.profile.preferences.get("llm", ""), r.profile.preferences
    assert r.episodic_count >= 20  # 10 user + 10 agent
    assert "Anthropic" in r.final_recall, r.final_recall
    assert r.recall_source == "semantic_profile", r.recall_source
    print("[OK] capstone_memory_chat._self_test passed (recall from turn 1 after 9 unrelated turns)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
