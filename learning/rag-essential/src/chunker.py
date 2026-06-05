"""Chunking strategies: fixed / overlap / sentence / semantic."""
from __future__ import annotations
from common import Chunk, hash_embed, cosine
import re


def fixed_chunk(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(1, size - overlap)
    return chunks


def sentence_chunk(text: str) -> list[str]:
    sents = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [s for s in sents if s]


def semantic_chunk(text: str, breakpoint_percentile: int = 50) -> list[str]:
    """Split where consecutive sentence similarity dips below threshold."""
    sents = sentence_chunk(text)
    if len(sents) < 2:
        return sents
    embeds = [hash_embed(s) for s in sents]
    sims = [cosine(embeds[i], embeds[i + 1]) for i in range(len(sents) - 1)]
    if not sims:
        return sents
    sorted_sims = sorted(sims)
    threshold = sorted_sims[len(sorted_sims) * breakpoint_percentile // 100]
    chunks = []
    cur = [sents[0]]
    for i, s in enumerate(sents[1:], start=1):
        if sims[i - 1] < threshold:
            chunks.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)
    chunks.append(" ".join(cur))
    return chunks


def to_chunk_objects(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [Chunk(doc_id=doc_id, chunk_id=f"{doc_id}_c{i}", text=t) for i, t in enumerate(texts)]


def _self_test() -> None:
    txt = "A" * 250
    cs = fixed_chunk(txt, size=100, overlap=20)
    assert len(cs) == 4, len(cs)
    assert all(len(c) <= 100 for c in cs)

    txt = "First sentence. Second one. Third! Fourth?"
    cs = sentence_chunk(txt)
    assert len(cs) == 4, cs

    long = "Cats are animals. Dogs are animals. Cars are vehicles. Bikes are vehicles. Apple is fruit. Banana is fruit."
    cs = semantic_chunk(long, breakpoint_percentile=50)
    assert len(cs) >= 1
    assert sum(len(c) for c in cs) >= len(long) - 5

    objs = to_chunk_objects("d1", ["chunk a", "chunk b"])
    assert len(objs) == 2 and objs[0].chunk_id == "d1_c0"
    print("[OK] chunker._self_test passed")


if __name__ == "__main__":
    _self_test()
