#!/usr/bin/env python3
"""Render one auditable dossier for core and direct-competitor papers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def evidence_lines(items: list[dict[str, object]], limit: int = 3) -> list[str]:
    if not items:
        return ["- No cue sentence extracted; inspect the reading card and full text."]
    return [f"- [{item['section']}, p. {item['page']}] {item['text']}" for item in items[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cards = [json.loads(line) for line in args.cards.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [card for card in cards if card["directness"] in {"core-selected", "direct-competitor"}]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for card in selected:
        groups[str(card["theme"])].append(card)

    lines = [
        "# Core and Direct-Competitor Full-Text Dossier", "",
        f"This dossier covers **{len(selected)}** papers: all screened core papers plus every full-corpus item classified as a direct competitor.",
        "Evidence is extracted from local full text and links back to a per-paper card and the source document.", "",
    ]
    for theme in sorted(groups):
        items = sorted(
            groups[theme],
            key=lambda card: (
                not bool(card["core_selected"]),
                -(max(int(detail["score"]) for detail in card["idea_relevance"].values())),
                int(card["all_rank"] or 99_999),
            ),
        )
        lines.extend([f"## {theme}", ""])
        for card in items:
            paper_id = str(card["paper_id"])
            warning = " **[WITHDRAWN]**" if card["withdrawn"] else ""
            lines.extend([
                f"### [{paper_id}: {card['title']}](../reading-cards/{paper_id}.md){warning}", "",
                f"- Class: `{card['directness']}`; pages: {card['pages']}; local text characters: {card['text_chars']}",
                f"- Full text: [{card['fulltext_kind']}]({card['fulltext_path']})",
                "- Five-idea scores: " + "; ".join(
                    f"{idea}={detail['score']}" for idea, detail in card["idea_relevance"].items()
                ),
                "", "**Contribution evidence**", "", *evidence_lines(card["contribution_evidence"]),
                "", "**Result evidence**", "", *evidence_lines(card["result_evidence"]),
                "", "**Limitations / caution evidence**", "", *evidence_lines(card["limitation_evidence"]), "",
            ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
