"""Minimal GraphRAG mock - entity extraction plus community detection plus summary."""
from __future__ import annotations
from common import Doc, tokenize, hash_embed, cosine, RetrievalResult, Chunk


KNOWN_ENTITIES = {
    "Anthropic", "Claude", "Microsoft", "Google", "OpenAI", "BAAI", "Voyage",
    "ReAct", "GraphRAG", "HippoRAG", "MCP", "BM25", "ColBERT", "HyDE",
    "RAGAS", "Self-RAG", "CRAG", "Cohere", "Multi-Query",
    "Louvain", "PageRank", "Robertson", "Khattab", "Yao", "Asai", "Yan",
    "BGE", "matryoshka",
}


def extract_entities(text: str) -> list[str]:
    found = []
    for ent in KNOWN_ENTITIES:
        if ent in text or ent.lower() in text.lower():
            found.append(ent)
    return found


class GraphRAGMock:
    def __init__(self):
        self.docs: list[Doc] = []
        self.entity_to_docs: dict[str, set[str]] = {}
        self.cooccur: dict[tuple[str, str], int] = {}
        self.communities: list[set[str]] = []
        self.community_summaries: list[str] = []

    def index(self, docs: list[Doc]) -> None:
        self.docs = docs
        for d in docs:
            ents = extract_entities(d.text)
            for e in ents:
                self.entity_to_docs.setdefault(e, set()).add(d.id)
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    key = tuple(sorted([e1, e2]))
                    self.cooccur[key] = self.cooccur.get(key, 0) + 1
        self._detect_communities()
        self._summarize_communities()

    def _detect_communities(self) -> None:
        """Mock community detection: simple connected components via co-occur edges."""
        adj: dict[str, set[str]] = {}
        for (e1, e2), w in self.cooccur.items():
            if w > 0:
                adj.setdefault(e1, set()).add(e2)
                adj.setdefault(e2, set()).add(e1)
        all_ents = set(self.entity_to_docs)
        seen: set[str] = set()
        for ent in all_ents:
            if ent in seen:
                continue
            comp = set()
            stack = [ent]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.add(cur)
                stack.extend(adj.get(cur, set()))
            if comp:
                self.communities.append(comp)

    def _summarize_communities(self) -> None:
        for comm in self.communities:
            related_docs = set()
            for e in comm:
                related_docs.update(self.entity_to_docs.get(e, set()))
            doc_texts = [d.text for d in self.docs if d.id in related_docs]
            summary = f"Community of {sorted(comm)[:5]}: " + " | ".join(doc_texts[:3])
            self.community_summaries.append(summary[:400])

    def query_local(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q_ents = extract_entities(query)
        matched_comm_idx = []
        for i, comm in enumerate(self.communities):
            if any(e in comm for e in q_ents):
                matched_comm_idx.append(i)

        if not matched_comm_idx:
            return self._fallback_search(query, k)

        results = []
        for idx in matched_comm_idx:
            summary = self.community_summaries[idx]
            ch = Chunk(doc_id=f"comm{idx}", chunk_id=f"comm{idx}", text=summary)
            results.append(RetrievalResult(chunk=ch, score=1.0))

        for d in self.docs:
            ents = extract_entities(d.text)
            if any(e in q_ents for e in ents):
                ch = Chunk(doc_id=d.id, chunk_id=d.id, text=d.text)
                results.append(RetrievalResult(chunk=ch, score=0.7))

        return results[:k]

    def _fallback_search(self, query: str, k: int) -> list[RetrievalResult]:
        q_vec = hash_embed(query)
        scored = []
        for d in self.docs:
            score = cosine(q_vec, hash_embed(d.text))
            ch = Chunk(doc_id=d.id, chunk_id=d.id, text=d.text)
            scored.append(RetrievalResult(chunk=ch, score=score))
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]

    def query_global(self, query: str) -> str:
        partials = []
        for i, summary in enumerate(self.community_summaries):
            partials.append(f"[comm{i}] {summary[:120]}")
        return f"Query: {query}\nPartials:\n" + "\n".join(partials)


def _self_test() -> None:
    from common import SAMPLE_DOCS

    g = GraphRAGMock()
    g.index(SAMPLE_DOCS)
    assert len(g.entity_to_docs) > 0
    assert "Anthropic" in g.entity_to_docs

    res = g.query_local("What did Anthropic build?", k=3)
    assert len(res) > 0
    text_blob = " ".join(r.chunk.text for r in res)
    assert "Anthropic" in text_blob, text_blob[:200]

    glob = g.query_global("Summarize all topics")
    assert "Partials:" in glob
    print("[OK] graph_rag._self_test passed")


if __name__ == "__main__":
    _self_test()
