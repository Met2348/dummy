"""HippoRAG mock — entity graph + personalized PageRank."""
from __future__ import annotations
from common import Doc, RetrievalResult, Chunk
from graph_rag import extract_entities


def personalized_pagerank(
    graph: dict[str, dict[str, float]],
    seeds: list[str],
    alpha: float = 0.15,
    iterations: int = 30,
) -> dict[str, float]:
    """Compute PPR with restart distribution on seeds. Standard formulation."""
    nodes = set(graph.keys())
    for adj in graph.values():
        nodes.update(adj)
    nodes = list(nodes)
    if not nodes:
        return {}
    valid_seeds = [s for s in seeds if s in nodes]
    if not valid_seeds:
        valid_seeds = nodes

    n = len(nodes)
    pr = {v: 1.0 / n for v in nodes}
    restart = {v: 1.0 / len(valid_seeds) if v in valid_seeds else 0.0 for v in nodes}

    for _ in range(iterations):
        new_pr = {v: alpha * restart[v] for v in nodes}
        for v in nodes:
            adj = graph.get(v, {})
            if not adj:
                continue
            total_w = sum(adj.values())
            if total_w == 0:
                continue
            for u, w in adj.items():
                new_pr[u] = new_pr.get(u, 0.0) + (1 - alpha) * pr[v] * (w / total_w)
        pr = new_pr
    return pr


class HippoRAG:
    def __init__(self):
        self.docs: list[Doc] = []
        self.entity_to_docs: dict[str, set[str]] = {}
        self.graph: dict[str, dict[str, float]] = {}

    def index(self, docs: list[Doc]) -> None:
        self.docs = docs
        for d in docs:
            ents = extract_entities(d.text)
            for e in ents:
                self.entity_to_docs.setdefault(e, set()).add(d.id)
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    self.graph.setdefault(e1, {})[e2] = self.graph.get(e1, {}).get(e2, 0) + 1
                    self.graph.setdefault(e2, {})[e1] = self.graph.get(e2, {}).get(e1, 0) + 1

    def search(self, query: str, k: int = 5, alpha: float = 0.15) -> list[RetrievalResult]:
        q_ents = extract_entities(query)
        if not q_ents:
            return []
        ppr = personalized_pagerank(self.graph, q_ents, alpha=alpha)
        ranked_ents = sorted(ppr, key=ppr.get, reverse=True)

        seen_docs: set[str] = set()
        results: list[RetrievalResult] = []
        doc_by_id = {d.id: d for d in self.docs}
        for ent in ranked_ents:
            for did in self.entity_to_docs.get(ent, set()):
                if did in seen_docs:
                    continue
                seen_docs.add(did)
                d = doc_by_id[did]
                ch = Chunk(doc_id=d.id, chunk_id=d.id, text=d.text)
                results.append(RetrievalResult(chunk=ch, score=ppr[ent]))
                if len(results) >= k:
                    return results
        return results


def _self_test() -> None:
    from common import SAMPLE_DOCS

    h = HippoRAG()
    h.index(SAMPLE_DOCS)
    assert "Anthropic" in h.graph or "Anthropic" in h.entity_to_docs

    res = h.search("Anthropic Claude MCP", k=3)
    assert len(res) > 0
    blob = " ".join(r.chunk.text for r in res)
    assert "Anthropic" in blob or "Claude" in blob or "MCP" in blob, blob
    print("[OK] hipporag._self_test passed")


if __name__ == "__main__":
    _self_test()
