#!/usr/bin/env python3
"""Create transport-specific evidence cards with page-located full-text snippets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    "transport_protocol": (
        "cross-model transfer", "cross model transfer", "cross-model generalization",
        "held-out model", "held out model", "unseen model", "source model", "target model",
        "model replacement", "model substitution", "additional model backbones",
        "transfer to 17 other models", "cross-backend", "backbone transfer",
    ),
    "prediction_and_sealing": (
        "predict", "prediction", "forecast", "prospective", "sealed", "leave-one-model-out",
        "unseen combination", "held-out cell", "calibration set", "out-of-sample",
    ),
    "atomicity_and_interaction": (
        "ablation", "component", "leave-one-layer-out", "factorial", "intervention",
        "treatment effect", "marginal effect", "interaction effect", "one variable",
        "controlled baseline", "ranking reversal", "rank reversal",
    ),
    "negative_transfer": (
        "negative transfer", "hurts", "degrades", "degraded", "collapse", "sign reversal",
        "worse than", "underperforms", "fails to transfer", "does not transfer", "non-monotone",
    ),
    "budget_and_operating_point": (
        "step budget", "token budget", "timeout", "cost-matched", "matched budget", "token cost",
        "operating point", "baseline success", "baseline performance", "source-tuned", "turn limit",
    ),
    "limitations_and_openings": (
        "limitation", "future work", "future direction", "remains open", "we leave", "not evaluated",
        "not study", "cannot conclude", "threats to validity", "generalizability",
    ),
}


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def collect(text: str, patterns: tuple[str, ...], limit: int = 8) -> list[dict[str, object]]:
    lowered = text.lower()
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        start = 0
        while True:
            index = lowered.find(pattern, start)
            if index < 0:
                break
            candidates.append((index, pattern))
            start = index + len(pattern)
    candidates.sort()
    rows: list[dict[str, object]] = []
    seen_pages: set[tuple[int, str]] = set()
    for index, pattern in candidates:
        page = text.count("\f", 0, index) + 1
        key = (page, pattern)
        if key in seen_pages:
            continue
        left = max(0, index - 340)
        right = min(len(text), index + len(pattern) + 620)
        snippet = compact(text[left:right])
        if len(snippet) > 950:
            snippet = snippet[:947].rstrip() + "..."
        rows.append({"page": page, "pattern": pattern, "snippet": snippet})
        seen_pages.add(key)
        if len(rows) >= limit:
            break
    return rows


def render(row: dict[str, object], groups: dict[str, list[dict[str, object]]]) -> str:
    paper_id = str(row["paper_id"])
    status = "withdrawn v2; archived v1 only" if row.get("withdrawn_current_version") else "active arXiv record"
    lines = [
        f"# {paper_id} - {row['title']}", "",
        f"- **Status:** {status}",
        f"- **Priority:** {row.get('curation_priority', '')}",
        f"- **Evidence role:** `{row.get('evidence_role', '')}`",
        f"- **Manual adjudication:** {row.get('adjudication', '')}",
        f"- **Published:** {row.get('published', '')}",
        f"- **PDF:** [local PDF](../papers/{paper_id}.pdf)",
        f"- **Full text:** [local text](../texts/{paper_id}.txt)",
        f"- **arXiv:** {row.get('abs_url', '')}", "",
        "## Author Abstract", "", str(row.get("abstract", "")), "",
    ]
    for group_name, evidence in groups.items():
        lines.extend([f"## {group_name.replace('_', ' ').title()}", ""])
        if not evidence:
            lines.append("- No explicit hit; absence is not evidence of absence.")
        else:
            for item in evidence:
                lines.append(f"- **p. {item['page']} | `{item['pattern']}`:** {item['snippet']}")
        lines.append("")
    lines.extend([
        "## Adjudication Checklist", "",
        "- [ ] Identify the exact transferred object.",
        "- [ ] Record source and target models/tasks.",
        "- [ ] Decide atomic versus bundled intervention.",
        "- [ ] Check whether the target was unseen during design or fitting.",
        "- [ ] Check whether target effect was predicted before observation.",
        "- [ ] Record budget, baseline, and operating-point controls.",
        "- [ ] Record positive, negative, mixed, or untested transfer.", "",
        "This card is a locator over extracted text. Verify tables, equations, and appendix details in the PDF before citing a numerical claim.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.cards_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Transport Evidence Cards", "",
        "Cards are ordered by manual priority and date. They locate evidence; they do not turn a phrase hit into a verified claim.", "",
        "| Priority | Paper | Role | Transport | Prediction | Atomicity | Negative | Budget |", "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        paper_id = str(row["paper_id"])
        text_path = args.texts_dir / f"{paper_id}.txt"
        if not text_path.exists():
            raise FileNotFoundError(text_path)
        text = text_path.read_text(encoding="utf-8", errors="replace").replace("\x00", " ")
        groups = {name: collect(text, patterns) for name, patterns in PATTERN_GROUPS.items()}
        (args.cards_dir / f"{paper_id}.md").write_text(render(row, groups), encoding="utf-8")
        counts = {name: len(items) for name, items in groups.items()}
        title = str(row["title"]).replace("|", "\\|")
        index_lines.append(
            f"| {row.get('curation_priority', '')} | [{paper_id}: {title}]({paper_id}.md) | "
            f"{row.get('evidence_role', '')} | {counts['transport_protocol']} | {counts['prediction_and_sealing']} | "
            f"{counts['atomicity_and_interaction']} | {counts['negative_transfer']} | {counts['budget_and_operating_point']} |"
        )
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"built {len(rows)} transport evidence cards")


if __name__ == "__main__":
    main()
