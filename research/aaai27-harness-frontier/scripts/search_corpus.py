#!/usr/bin/env python3
"""Search the local full-text paper corpus with SQLite FTS5."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="FTS5 query, for example: 'harness AND transport' or 'prequential'")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--literal", action="store_true", help="Treat the query as one literal phrase")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = f'"{args.query.replace(chr(34), chr(34) * 2)}"' if args.literal else args.query

    connection = sqlite3.connect(args.database)
    try:
        rows = connection.execute(
            "SELECT paper_id, title, bm25(papers_fts, 0.0, 3.0, 1.0, 1.0) AS score, "
            "snippet(papers_fts, 3, '[', ']', ' ... ', 36) AS evidence "
            "FROM papers_fts WHERE papers_fts MATCH ? ORDER BY score LIMIT ?",
            (query, args.limit),
        ).fetchall()
    finally:
        connection.close()

    for rank, (paper_id, title, score, evidence) in enumerate(rows, start=1):
        print(f"{rank:02d}. {paper_id} | {score:.4f} | {title}")
        print(f"    {evidence}")


if __name__ == "__main__":
    main()
