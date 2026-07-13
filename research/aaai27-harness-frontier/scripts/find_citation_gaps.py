#!/usr/bin/env python3
"""Find cited arXiv papers that are absent from the current corpus manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ARXIV_RE = re.compile(r"(?:arxiv\s*(?::|preprint)?\s*)?(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
RELEVANCE_RE = re.compile(
    r"\b(harness|scaffold|agent|agentic|orchestrat|self-evol|evaluation|benchmark|"
    r"transport|transfer|robust|reliab|causal|intervention|bayesian|configuration|"
    r"research|scientist|tool|context|memory|verification)\w*\b",
    re.I,
)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {str(item["paper_id"]) for item in manifest}
    cards = [json.loads(line) for line in args.cards.read_text(encoding="utf-8").splitlines() if line.strip()]
    card_by_id = {str(card["paper_id"]): card for card in cards}

    citing: dict[str, set[str]] = defaultdict(set)
    contexts: dict[str, list[dict[str, str]]] = defaultdict(list)
    relevance_hits: dict[str, int] = defaultdict(int)
    core_citers: dict[str, set[str]] = defaultdict(set)
    direct_citers: dict[str, set[str]] = defaultdict(set)

    for text_path in sorted(args.texts_dir.glob("*.txt")):
        citing_id = text_path.stem
        raw = text_path.read_text(encoding="utf-8", errors="replace")
        for match in ARXIV_RE.finditer(raw):
            cited_id = match.group(1)
            if cited_id in known or cited_id == citing_id:
                continue
            citing[cited_id].add(citing_id)
            start = max(0, match.start() - 280)
            end = min(len(raw), match.end() + 280)
            snippet = compact(raw[start:end])
            hits = len(RELEVANCE_RE.findall(snippet))
            relevance_hits[cited_id] += hits
            card = card_by_id.get(citing_id, {})
            if card.get("core_selected"):
                core_citers[cited_id].add(citing_id)
            if card.get("directness") == "direct-competitor":
                direct_citers[cited_id].add(citing_id)
            if len(contexts[cited_id]) < 5:
                contexts[cited_id].append({"citing_paper": citing_id, "snippet": snippet})

    rows: list[dict[str, object]] = []
    for cited_id, citing_ids in citing.items():
        year = int("20" + cited_id[:2])
        recency = 5 if year >= 2026 else 3 if year >= 2025 else 1 if year >= 2023 else 0
        score = (
            len(citing_ids)
            + 9 * len(core_citers[cited_id])
            + 7 * len(direct_citers[cited_id])
            + min(relevance_hits[cited_id], 30)
            + recency
        )
        rows.append(
            {
                "paper_id": cited_id,
                "citation_gap_score": score,
                "citing_papers": sorted(citing_ids),
                "citation_count": len(citing_ids),
                "core_citers": sorted(core_citers[cited_id]),
                "direct_citers": sorted(direct_citers[cited_id]),
                "relevance_context_hits": relevance_hits[cited_id],
                "contexts": contexts[cited_id],
            }
        )
    rows.sort(key=lambda row: (int(row["citation_gap_score"]), int(row["citation_count"])), reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    lines = [
        "# Citation-Chasing Gaps", "",
        f"Found **{len(rows)}** arXiv identifiers cited by the corpus but absent from the manifest.",
        "This is a triage list; references become corpus papers only after metadata relevance screening.", "",
        "| Rank | arXiv | Score | Citers | Core citers | Direct citers | Example context |", "|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:250], start=1):
        context = str(row["contexts"][0]["snippet"] if row["contexts"] else "").replace("|", "\\|")[:280]
        lines.append(
            f"| {rank} | [{row['paper_id']}](https://arxiv.org/abs/{row['paper_id']}) | {row['citation_gap_score']} | "
            f"{row['citation_count']} | {len(row['core_citers'])} | {len(row['direct_citers'])} | {context} |"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} citation gaps")


if __name__ == "__main__":
    main()
