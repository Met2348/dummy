"""Semantic memory — KG triples + vector store."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import hash_embed, cosine


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    user_id: str
    embedding: list[float] = field(default_factory=list)


class SemanticMemory:
    def __init__(self):
        self.triples: list[Triple] = []
        self.graph: dict[str, dict[str, list[str]]] = {}

    def add_triple(self, subj: str, pred: str, obj: str, user_id: str) -> Triple:
        existing = self.graph.setdefault(subj, {}).setdefault(pred, [])
        if obj in existing:
            return next(t for t in self.triples
                        if t.subject == subj and t.predicate == pred and t.object == obj
                        and t.user_id == user_id)
        existing.append(obj)
        triple = Triple(
            subject=subj, predicate=pred, object=obj, user_id=user_id,
            embedding=hash_embed(f"{subj} {pred} {obj}"),
        )
        self.triples.append(triple)
        return triple

    def update_triple(self, subj: str, pred: str, new_obj: str, user_id: str) -> bool:
        existing = self.graph.get(subj, {}).get(pred, [])
        for t in self.triples:
            if (t.subject == subj and t.predicate == pred
                    and t.user_id == user_id and t.object in existing):
                old_obj = t.object
                t.object = new_obj
                t.embedding = hash_embed(f"{subj} {pred} {new_obj}")
                self.graph[subj][pred] = [
                    new_obj if o == old_obj else o
                    for o in self.graph[subj][pred]
                ]
                return True
        self.add_triple(subj, pred, new_obj, user_id)
        return False

    def query(self, subj: str, pred: str | None = None):
        if pred is None:
            return self.graph.get(subj, {})
        return self.graph.get(subj, {}).get(pred, [])

    def search(self, query: str, user_id: str | None = None, k: int = 5) -> list[Triple]:
        q_vec = hash_embed(query)
        scored = []
        for t in self.triples:
            if user_id is not None and t.user_id != user_id:
                continue
            scored.append((t, cosine(q_vec, t.embedding)))
        return [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]

    def multi_hop(self, subj: str, pred1: str, pred2: str) -> list[str]:
        targets = []
        for intermediate in self.query(subj, pred1):
            targets.extend(self.query(intermediate, pred2))
        return targets


def _self_test() -> None:
    mem = SemanticMemory()
    mem.add_triple("Alice", "is_a", "ML Engineer", "alice")
    mem.add_triple("Alice", "prefers", "Anthropic Claude", "alice")
    mem.add_triple("Alice", "works_on", "RAG_legal", "alice")
    mem.add_triple("RAG_legal", "uses_framework", "LangChain", "alice")

    pref = mem.query("Alice", "prefers")
    assert pref == ["Anthropic Claude"], pref

    updated = mem.update_triple("Alice", "prefers", "Gemini", "alice")
    assert updated is True
    assert mem.query("Alice", "prefers") == ["Gemini"]

    frameworks = mem.multi_hop("Alice", "works_on", "uses_framework")
    assert "LangChain" in frameworks, frameworks

    found = mem.search("Alice preference", user_id="alice", k=2)
    assert len(found) > 0
    print("[OK] semantic_memory._self_test passed")


if __name__ == "__main__":
    _self_test()
