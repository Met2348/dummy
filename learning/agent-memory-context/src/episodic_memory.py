"""Episodic memory: time-stamped event store with user filter."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import hash_embed, cosine, mock_now


@dataclass
class Episode:
    id: str
    user_id: str
    timestamp: float
    actor: str
    content: str
    embedding: list[float] = field(default_factory=list)


class EpisodicMemory:
    def __init__(self):
        self.episodes: list[Episode] = []

    def add(self, user_id: str, actor: str, content: str) -> Episode:
        ep = Episode(
            id=f"ep{len(self.episodes)}",
            user_id=user_id,
            timestamp=mock_now(),
            actor=actor,
            content=content,
            embedding=hash_embed(content),
        )
        self.episodes.append(ep)
        return ep

    def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> list[Episode]:
        q_vec = hash_embed(query)
        scored: list[tuple[Episode, float]] = []
        for ep in self.episodes:
            if ep.user_id != user_id:
                continue
            if time_from is not None and ep.timestamp < time_from:
                continue
            if time_to is not None and ep.timestamp > time_to:
                continue
            scored.append((ep, cosine(q_vec, ep.embedding)))
        return [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]

    def get_by_user(self, user_id: str) -> list[Episode]:
        return [e for e in self.episodes if e.user_id == user_id]


def _self_test() -> None:
    from common import reset_mock_time

    reset_mock_time()
    mem = EpisodicMemory()
    mem.add("alice", "user", "What is RAG?")
    t_middle = mock_now()
    mem.add("alice", "user", "Tell me about ColBERT late interaction")
    mem.add("alice", "user", "How does GraphRAG work?")
    mem.add("bob", "user", "What's the weather?")

    found = mem.search("ColBERT MaxSim", user_id="alice", k=3)
    assert len(found) > 0
    assert any("ColBERT" in e.content for e in found), [e.content for e in found]

    bob_only = mem.search("weather", user_id="bob", k=5)
    assert len(bob_only) == 1
    assert "weather" in bob_only[0].content

    recent = mem.search("RAG", user_id="alice", k=5, time_from=t_middle)
    assert all(e.timestamp >= t_middle for e in recent)
    print("[OK] episodic_memory._self_test passed")


if __name__ == "__main__":
    _self_test()
