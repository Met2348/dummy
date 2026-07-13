#!/usr/bin/env python3
"""Score citation-chased metadata for relevance to harness and AI-research work."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from search_arxiv import score


EXPLICIT_RE = re.compile(
    r"\b(harness|scaffold|agent|agentic|ai scientist|automated research|research agent|"
    r"scientific discovery|paperbench|swe-bench|terminal-bench)\b",
    re.I,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-relevance", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: list[dict[str, object]] = []
    for row in rows:
        relevance, reasons = score(row)
        blob = f"{row.get('title', '')} {row.get('abstract', '')}"
        row["relevance_score"] = relevance
        row["score_reasons"] = reasons
        row["citation_screen_score"] = relevance + min(int(row.get("citation_gap_score", 0)), 40)
        if relevance >= args.min_relevance and EXPLICIT_RE.search(blob):
            row["source_pools"] = ["citation_chasing_2026"]
            row["matched_queries"] = ["full-corpus citation chasing"]
            selected.append(row)
    selected.sort(
        key=lambda row: (int(row["citation_screen_score"]), int(row["relevance_score"])), reverse=True
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    lines = [
        "# Screened 2026 Citation-Chasing Additions", "",
        f"Metadata fetched: **{len(rows)}**; relevance-screened additions: **{len(selected)}**.", "",
        "| Rank | arXiv | Title | Relevance | Citation score | Combined | Reasons |", "|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(selected, start=1):
        title = str(row["title"]).replace("|", "\\|")
        lines.append(
            f"| {rank} | [{row['paper_id']}]({row['abs_url']}) | {title} | {row['relevance_score']} | "
            f"{row.get('citation_gap_score', 0)} | {row['citation_screen_score']} | {', '.join(row['score_reasons'])} |"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"screened {len(rows)} metadata records -> {len(selected)} additions")


if __name__ == "__main__":
    main()
