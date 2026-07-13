#!/usr/bin/env python3
"""Render the selected-paper manifest as a readable Markdown index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = Counter(str(row["category"]) for row in rows)
    lines = [
        "# Core Library Index",
        "",
        f"Curated papers: **{len(rows)}**. Search cutoff: **2026-07-10**.",
        "",
        "The list is ordered by research role, not by citation count. `A` means central to",
        "idea selection; `B` means useful context or a methodological bridge.",
        "",
        "## Category counts",
        "",
        "| Category | Papers |",
        "|---|---:|",
    ]
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "## Papers",
            "",
            "| # | Pri. | Category | Year | Paper | Local PDF |",
            "|---:|:---:|---|:---:|---|---|",
        ]
    )
    for row in rows:
        paper_id = str(row["paper_id"])
        title = str(row["title"]).replace("|", "\\|")
        year = str(row["published"])[:4]
        lines.append(
            f'| {row["selection_rank"]} | {row["priority"]} | {row["category"]} | {year} | '
            f'[{title}]({row["abs_url"]}) | [PDF](../papers/{paper_id}.pdf) |'
        )
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

