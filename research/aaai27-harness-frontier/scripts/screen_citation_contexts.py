#!/usr/bin/env python3
"""Screen citation gaps by idea-specific wording in their citation contexts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS: dict[str, tuple[str, ...]] = {
    "PREQ-Harness": (
        "prequential", "anytime-valid", "confidence sequence", "reusable holdout",
        "forward generalization", "adaptive evaluation",
    ),
    "Harness Transport": (
        "harness transfer", "cross-model transfer", "cross model transfer", "transportability",
        "held-out model", "model upgrade", "model-specific harness",
    ),
    "MRT-Harness": (
        "micro-randomized", "micro randomized", "sequential intervention", "causal excursion",
        "time-varying treatment",
    ),
    "Harness-C": (
        "harness corruption", "context corruption", "fault injection", "stability envelope",
        "controller corruption", "harness robustness", "perturbation benchmark",
    ),
    "ActiveHarness": (
        "harness optimization", "harness selection", "bayesian optimization", "best-arm",
        "best arm", "configuration search", "successive halving", "sample-efficient search",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gaps = [json.loads(line) for line in args.gaps.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for gap in gaps:
        blob = " ".join(str(item.get("snippet", "")) for item in gap.get("contexts", [])).lower()
        idea_hits: dict[str, list[str]] = {}
        for idea, patterns in PATTERNS.items():
            hits = [pattern for pattern in patterns if pattern in blob]
            if hits:
                idea_hits[idea] = hits
        if not idea_hits:
            continue
        gap["idea_context_hits"] = idea_hits
        gap["idea_context_score"] = (
            12 * sum(len(hits) for hits in idea_hits.values())
            + 4 * len(gap.get("core_citers", []))
            + 3 * len(gap.get("direct_citers", []))
            + min(int(gap.get("citation_count", 0)), 10)
        )
        rows.append(gap)
    rows.sort(key=lambda row: int(row["idea_context_score"]), reverse=True)

    lines = [
        "# Idea-Specific Citation-Gap Screen", "",
        f"Idea-specific wording found for **{len(rows)}** missing arXiv references.", "",
        "| Rank | arXiv | Score | Ideas / phrases | Citers | Context |", "|---:|---|---:|---|---:|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        ideas = "; ".join(f"{idea}: {', '.join(hits)}" for idea, hits in row["idea_context_hits"].items())
        context = str(row["contexts"][0]["snippet"] if row.get("contexts") else "").replace("|", "\\|")[:360]
        lines.append(
            f"| {rank} | [{row['paper_id']}](https://arxiv.org/abs/{row['paper_id']}) | "
            f"{row['idea_context_score']} | {ideas} | {row['citation_count']} | {context} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"screened {len(gaps)} gaps -> {len(rows)} idea-specific candidates")


if __name__ == "__main__":
    main()
