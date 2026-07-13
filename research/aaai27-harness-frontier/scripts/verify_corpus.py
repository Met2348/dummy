#!/usr/bin/env python3
"""Verify that every manifest record has a readable local full-text representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["paper_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    parser.add_argument("--withdrawn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_ids = load_jsonl_ids(args.manifest)
    pdf_paths = sorted(args.papers_dir.glob("*.pdf"))
    pdf_ids = {path.stem for path in pdf_paths}
    text_ids = {path.stem for path in args.texts_dir.glob("*.txt")}
    card_ids = {path.stem for path in args.cards_dir.glob("*.md")}
    withdrawn = json.loads(args.withdrawn.read_text(encoding="utf-8"))
    html_only_ids = {
        str(row["paper_id"])
        for row in withdrawn
        if str(row.get("local_fulltext_path", "")).lower().endswith(".html")
    }

    invalid_magic: list[str] = []
    for path in pdf_paths:
        with path.open("rb") as handle:
            if handle.read(4) != b"%PDF":
                invalid_magic.append(path.stem)

    missing_pdf_or_html = sorted(manifest_ids - pdf_ids - html_only_ids)
    status = {
        "manifest_records": len(manifest_ids),
        "pdf_files": len(pdf_ids),
        "html_only_fulltexts": sorted(html_only_ids),
        "text_files": len(text_ids),
        "reading_cards": len(card_ids),
        "pdf_bytes": sum(path.stat().st_size for path in pdf_paths),
        "invalid_pdf_magic": invalid_magic,
        "missing_pdf_or_html": missing_pdf_or_html,
        "missing_text": sorted(manifest_ids - text_ids),
        "missing_card": sorted(manifest_ids - card_ids),
        "orphan_pdf": sorted(pdf_ids - manifest_ids),
        "orphan_text": sorted(text_ids - manifest_ids),
        "orphan_card": sorted(card_ids - manifest_ids),
    }
    status["complete"] = not any(
        status[key]
        for key in ("invalid_pdf_magic", "missing_pdf_or_html", "missing_text", "missing_card")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
