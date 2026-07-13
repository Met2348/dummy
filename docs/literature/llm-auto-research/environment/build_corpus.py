from __future__ import annotations

import argparse
from pathlib import Path

from lib_corpus import build_records, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local JSONL corpus from the paper package.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    out = args.out or args.root / "metadata" / "corpus.jsonl"
    records = build_records(args.root, max_chars=args.max_chars, overlap=args.overlap)
    write_jsonl(out, records)
    paper_count = len({record["paper_id"] for record in records})
    print(f"wrote {len(records)} chunks from {paper_count} papers to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
