"""LlamaIndex-style RAG mock."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import mock_search, mock_summarize


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Node:
    doc_id: str
    node_id: str
    text: str


class VectorStoreIndex:
    def __init__(self):
        self.nodes: list[Node] = []
        self.documents: list[Document] = []

    @classmethod
    def from_documents(cls, docs: list[Document]) -> "VectorStoreIndex":
        idx = cls()
        idx.documents = list(docs)
        for d in docs:
            chunks = [d.text[i:i + 100] for i in range(0, len(d.text), 100)]
            for i, c in enumerate(chunks):
                idx.nodes.append(Node(doc_id=d.id, node_id=f"{d.id}_n{i}", text=c))
        return idx

    def as_query_engine(self, k: int = 3) -> "QueryEngine":
        return QueryEngine(self, k=k)


class QueryEngine:
    def __init__(self, index: VectorStoreIndex, k: int = 3):
        self.index = index
        self.k = k

    def query(self, query: str) -> dict:
        contexts = mock_search(query, k=self.k)
        for node in self.index.nodes:
            if any(q in node.text.lower() for q in query.lower().split()):
                contexts.append(node.text)
        contexts = contexts[: self.k]
        return {
            "response": mock_summarize(query, contexts),
            "source_nodes": [c[:60] for c in contexts],
        }


def _self_test() -> None:
    docs = [
        Document(id="d1", text="ReAct is a Thought-Action-Observation loop. " * 5),
        Document(id="d2", text="GraphRAG uses entity graph and community summary. " * 5),
    ]
    idx = VectorStoreIndex.from_documents(docs)
    assert idx.documents == docs
    assert len(idx.nodes) >= 2

    qe = idx.as_query_engine(k=3)
    result = qe.query("What is ReAct?")
    assert "Summary about" in result["response"]
    assert len(result["source_nodes"]) >= 1
    print("[OK] llamaindex_style._self_test passed")


if __name__ == "__main__":
    _self_test()
