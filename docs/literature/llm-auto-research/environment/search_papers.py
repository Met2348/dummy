from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from lib_corpus import read_jsonl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def score_record(query_terms: Counter, record: dict) -> float:
    text_terms = Counter(tokenize(record.get("text", "")))
    title_terms = Counter(tokenize(record.get("title", "")))
    if not text_terms:
        return 0.0
    score = 0.0
    for term, q_count in query_terms.items():
        score += q_count * text_terms.get(term, 0)
        score += 3.0 * q_count * title_terms.get(term, 0)
    return score / math.sqrt(sum(text_terms.values()))


def make_snippet(text: str, query_terms: Counter, width: int = 280) -> str:
    lower = text.lower()
    positions = [lower.find(term) for term in query_terms if lower.find(term) >= 0]
    start = max(min(positions) - 80, 0) if positions else 0
    snippet = text[start : start + width].replace("\n", " ").strip()
    return snippet + ("..." if start + width < len(text) else "")


def search_records(query: str, records: list[dict], top_k: int = 8) -> list[dict]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return []
    ranked = []
    for record in records:
        score = score_record(query_terms, record)
        if score <= 0:
            continue
        result = dict(record)
        result["score"] = round(score, 4)
        result["snippet"] = make_snippet(record.get("text", ""), query_terms)
        ranked.append(result)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def format_result(result: dict) -> str:
    return (
        f"[{result['score']:.4f}] {result['paper_id']} {result['title']} "
        f"(chunk {result['chunk_id']})\n  {result['snippet']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the local LLM Auto Research corpus.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    corpus = args.corpus or args.root / "metadata" / "corpus.jsonl"
    records = read_jsonl(corpus)
    for result in search_records(" ".join(args.query), records, top_k=args.top_k):
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
