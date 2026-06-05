"""Minimal vector store with metadata filter."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from common import cosine


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)


class SimpleVectorStore:
    def __init__(self):
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, id: str, vector: list[float], payload: dict | None = None) -> None:
        self.records[id] = VectorRecord(id=id, vector=vector, payload=payload or {})

    def delete(self, id: str) -> bool:
        return self.records.pop(id, None) is not None

    def search(
        self,
        query_vec: list[float],
        k: int = 5,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        candidates = []
        for rec in self.records.values():
            if filter_fn is not None and not filter_fn(rec.payload):
                continue
            score = cosine(query_vec, rec.vector)
            candidates.append((rec, score))
        return sorted(candidates, key=lambda r: r[1], reverse=True)[:k]

    def size(self) -> int:
        return len(self.records)


def _self_test() -> None:
    from common import hash_embed

    store = SimpleVectorStore()
    store.upsert("r1", hash_embed("Anthropic Claude AI"), {"user": "alice"})
    store.upsert("r2", hash_embed("OpenAI GPT"), {"user": "alice"})
    store.upsert("r3", hash_embed("Anthropic Claude AI"), {"user": "bob"})
    assert store.size() == 3

    results = store.search(hash_embed("Claude"), k=2)
    assert len(results) == 2
    top_ids = {r.id for r, _ in results}
    assert "r1" in top_ids or "r3" in top_ids

    results = store.search(
        hash_embed("Claude"), k=5, filter_fn=lambda p: p.get("user") == "alice"
    )
    assert all(r.payload["user"] == "alice" for r, _ in results)

    assert store.delete("r1") is True
    assert store.delete("nonexistent") is False
    assert store.size() == 2
    print("[OK] vector_store._self_test passed")


if __name__ == "__main__":
    _self_test()
