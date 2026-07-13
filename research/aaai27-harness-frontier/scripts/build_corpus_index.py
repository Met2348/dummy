#!/usr/bin/env python3
"""Build a local SQLite FTS5 index over every paper's full text."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    connection = sqlite3.connect(args.output)
    try:
        connection.execute(
            "CREATE TABLE papers (paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, published TEXT, "
            "core_selected INTEGER, full_text TEXT)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE papers_fts USING fts5(paper_id UNINDEXED, title, abstract, full_text, "
            "tokenize='porter unicode61')"
        )
        for index, record in enumerate(records, start=1):
            paper_id = str(record["paper_id"])
            text_path = args.texts_dir / f"{paper_id}.txt"
            full_text = text_path.read_text(encoding="utf-8", errors="replace").replace("\x00", " ")
            values = (
                paper_id, str(record.get("title", "")), str(record.get("abstract", "")),
                str(record.get("published", "")), int(bool(record.get("core_selected"))), full_text,
            )
            connection.execute("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)", values)
            connection.execute(
                "INSERT INTO papers_fts(paper_id, title, abstract, full_text) VALUES (?, ?, ?, ?)",
                (values[0], values[1], values[2], values[5]),
            )
            if index % 100 == 0 or index == len(records):
                print(f"[{index:04d}/{len(records):04d}] indexed", flush=True)
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
