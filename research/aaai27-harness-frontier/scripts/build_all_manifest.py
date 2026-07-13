#!/usr/bin/env python3
"""Merge broad and narrow arXiv pools into one deduplicated manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    selected_ids = {str(row["paper_id"]) for row in selected}
    merged: dict[str, dict[str, object]] = {}
    source_counts: dict[str, int] = {}

    for source in args.input:
        rows = load_jsonl(source)
        source_counts[str(source)] = len(rows)
        for row in rows:
            paper_id = str(row.get("arxiv_id") or row.get("paper_id"))
            normalized = dict(row)
            normalized["paper_id"] = paper_id
            normalized["pdf_url"] = str(row.get("pdf_url") or f"https://arxiv.org/pdf/{paper_id}")
            normalized["abs_url"] = str(row.get("abs_url") or f"https://arxiv.org/abs/{paper_id}")
            normalized["core_selected"] = paper_id in selected_ids or bool(row.get("core_selected"))
            normalized["source_pools"] = [str(source)]
            if paper_id not in merged:
                merged[paper_id] = normalized
                continue

            current = merged[paper_id]
            current["source_pools"] = sorted(
                set(list(current.get("source_pools", [])) + [str(source)])
            )
            current["matched_queries"] = sorted(
                set(list(current.get("matched_queries", [])) + list(row.get("matched_queries", [])))
            )
            current["relevance_score"] = max(
                int(current.get("relevance_score", 0)), int(row.get("relevance_score", 0))
            )
            current["core_selected"] = (
                paper_id in selected_ids
                or bool(current.get("core_selected"))
                or bool(row.get("core_selected"))
            )

    for row in selected:
        paper_id = str(row["paper_id"])
        if paper_id in merged:
            merged[paper_id]["core_selected"] = True
            continue
        normalized = dict(row)
        normalized["paper_id"] = paper_id
        normalized["pdf_url"] = str(row.get("pdf_url") or f"https://arxiv.org/pdf/{paper_id}")
        normalized["abs_url"] = str(row.get("abs_url") or f"https://arxiv.org/abs/{paper_id}")
        normalized["core_selected"] = True
        normalized["source_pools"] = [str(args.selected)]
        normalized["matched_queries"] = []
        normalized["relevance_score"] = int(row.get("relevance_score", 0))
        merged[paper_id] = normalized

    rows = sorted(
        merged.values(),
        key=lambda row: (
            bool(row.get("core_selected")),
            int(row.get("relevance_score", 0)),
            str(row.get("published", "")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["all_rank"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "source_counts": source_counts,
        "unique_papers": len(rows),
        "core_selected_present": sum(bool(row.get("core_selected")) for row in rows),
    }
    args.output.with_name("all_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
