#!/usr/bin/env python3
"""Hard-link selected transport papers, texts, and cards into the isolated project."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def materialize(source: Path, target: Path) -> str:
    if target.exists():
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-papers", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--source-cards", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        row for index, row in enumerate(rows, start=1)
        if bool(row.get("deep_read"))
        or int(row.get("transport_rank", index)) <= args.top
        or int(row.get("curation_priority", 99)) <= 3
    ]
    status: list[dict[str, object]] = []
    for row in selected:
        paper_id = str(row["paper_id"])
        item: dict[str, object] = {
            "paper_id": paper_id,
            "rank": row.get("transport_rank"),
            "curation_priority": row.get("curation_priority"),
        }
        pdf_source = args.source_papers / f"{paper_id}.pdf"
        if pdf_source.exists():
            item["pdf"] = materialize(pdf_source, args.project_root / "papers" / f"{paper_id}.pdf")
        else:
            item["pdf"] = "missing_source"
        text_source = args.source_texts / f"{paper_id}.txt"
        card_source = args.source_cards / f"{paper_id}.md"
        item["text"] = (
            materialize(text_source, args.project_root / "texts" / f"{paper_id}.txt")
            if text_source.exists() else "missing_source"
        )
        item["card"] = (
            materialize(card_source, args.project_root / "reading-cards" / f"{paper_id}.md")
            if card_source.exists() else "missing_source"
        )
        status.append(item)

    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"materialized {len(status)} transport papers")


if __name__ == "__main__":
    main()
