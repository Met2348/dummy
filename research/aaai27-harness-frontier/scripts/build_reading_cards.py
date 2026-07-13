#!/usr/bin/env python3
"""Create traceable, per-paper reading cards from the downloaded full-text corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


IDEA_PATTERNS: dict[str, tuple[str, ...]] = {
    "PREQ-Harness": (
        "prequential", "anytime-valid", "confidence sequence", "e-process",
        "adaptive data analysis", "reusable holdout", "forward generalization",
        "adaptive evaluation", "test-time adaptation", "self-evolving",
        "harness evolution", "promotion gate", "online evaluation",
    ),
    "Harness Transport": (
        "transportability", "cross-model transfer", "cross model transfer",
        "held-out model", "unseen model", "model upgrade", "model-specific",
        "backbone transfer", "transfer matrix", "portability", "ranking reversal",
        "model-harness interaction", "model harness interaction",
    ),
    "MRT-Harness": (
        "micro-randomized", "micro randomized", "sequential intervention",
        "time-varying treatment", "causal excursion", "proximal effect",
        "causal attribution", "sequential causal", "randomized intervention",
        "dynamic treatment", "off-policy evaluation",
    ),
    "Harness-C": (
        "common corruption", "corruption benchmark", "stability envelope",
        "perturbation", "stress test", "stress-test", "robustness",
        "fault injection", "context corruption", "tool failure", "error recovery",
        "distribution shift", "adversarial environment",
    ),
    "ActiveHarness": (
        "active selection", "best-arm", "best arm", "sample-efficient selection",
        "bayesian optimization", "bandit", "combinatorial optimization",
        "successive halving", "racing algorithm", "active testing",
        "adaptive experiment", "configuration search",
    ),
}

THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "measurement-evaluation": (
        "evaluation", "benchmark", "metric", "variance decomposition", "leaderboard",
        "confidence interval", "statistical significance", "judge", "grading",
    ),
    "optimization-evolution": (
        "evolution", "self-improving", "self-evolving", "optimization", "search",
        "genetic", "reinforcement learning", "meta-agent", "automated design",
    ),
    "transfer-generalization": (
        "transfer", "generalization", "held-out", "unseen model", "portability",
        "cross-model", "domain shift", "model-specific",
    ),
    "sequential-causal": (
        "causal", "intervention", "treatment effect", "counterfactual", "sequential",
        "off-policy", "randomized", "attribution",
    ),
    "robustness-reliability": (
        "robustness", "reliability", "failure", "recovery", "perturbation",
        "corruption", "fault", "stability", "safety",
    ),
    "context-memory-tools": (
        "context management", "memory", "tool use", "tool-use", "tool calling",
        "retrieval", "state", "planning", "verification", "orchestration",
    ),
    "automated-research": (
        "ai scientist", "automated research", "scientific discovery", "research agent",
        "paperbench", "research idea", "experiment design", "hypothesis generation",
        "literature review", "research reproduction",
    ),
}

SECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abstract", ("abstract",)),
    ("introduction", ("introduction", "background and motivation")),
    ("related_work", ("related work", "background", "preliminaries")),
    ("method", ("method", "methods", "methodology", "approach", "framework", "system design")),
    ("experiments", ("experiments", "experimental setup", "evaluation", "empirical evaluation")),
    ("results", ("results", "main results", "analysis")),
    ("limitations", ("limitations", "limitation", "threats to validity", "discussion and limitations")),
    ("conclusion", ("conclusion", "conclusions", "concluding remarks", "discussion and conclusion")),
)

METHOD_CUES = (
    "we propose", "we introduce", "we present", "we develop", "we design",
    "our method", "our framework", "our approach", "consists of", "is composed of",
)
RESULT_CUES = (
    "we find", "we found", "we observe", "we show", "we demonstrate", "results show",
    "outperform", "improve", "achieve", "significant", "our results", "experiments show",
)


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\u00ad", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def heading_label(line: str) -> str | None:
    candidate = compact(line).lstrip("#").strip()
    if not candidate or len(candidate) > 110:
        return None
    candidate = re.sub(r"^(?:section\s+)?(?:\d+(?:\.\d+)*|[ivx]+)[\s.:\-)]+", "", candidate, flags=re.I)
    lowered = candidate.lower().strip(" .:-")
    for label, aliases in SECTION_ALIASES:
        if any(lowered == alias or lowered.startswith(alias + ":") or lowered.startswith(alias + " ") for alias in aliases):
            return label
    return None


def section_map(raw: str) -> dict[str, dict[str, object]]:
    lines = raw.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    offset = 0
    for index, line in enumerate(lines):
        label = heading_label(line)
        if label:
            headings.append((index, offset, label))
        offset += len(line)

    sections: dict[str, dict[str, object]] = {}
    for position, (line_index, char_offset, label) in enumerate(headings):
        if label in sections:
            continue
        next_line = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = clean_text("".join(lines[line_index + 1 : next_line]))
        if len(body) < 80:
            continue
        page = raw.count("\f", 0, char_offset) + 1
        sections[label] = {"page": page, "text": body[:8_000]}
    return sections


def sentences(value: str) -> list[str]:
    value = compact(value)
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
    output: list[str] = []
    seen: set[str] = set()
    for sentence in candidates:
        sentence = compact(sentence)
        if len(sentence) < 55 or len(sentence) > 700:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(sentence)
    return output


def select_evidence(
    sections: dict[str, dict[str, object]],
    section_order: Iterable[str],
    cues: tuple[str, ...] | None,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for section_name in section_order:
        section = sections.get(section_name)
        if not section:
            continue
        for sentence in sentences(str(section["text"])):
            lowered = sentence.lower()
            if cues and not any(cue in lowered for cue in cues):
                continue
            snippet = sentence[:300].rstrip()
            if len(sentence) > 300:
                snippet += "..."
            selected.append({"section": section_name, "page": section["page"], "text": snippet})
            if len(selected) >= limit:
                return selected
    return selected


def pattern_score(title_abstract: str, full_text: str, patterns: tuple[str, ...]) -> tuple[int, dict[str, int]]:
    title_abstract = title_abstract.lower()
    full_text = full_text.lower()
    hits: dict[str, int] = {}
    score = 0
    for pattern in patterns:
        lead_count = title_abstract.count(pattern)
        full_count = full_text.count(pattern)
        if full_count:
            hits[pattern] = full_count
            score += min(full_count, 8) + (4 * lead_count)
    return score, hits


def best_theme(title_abstract: str, full_text: str) -> tuple[str, dict[str, int]]:
    scores: dict[str, int] = {}
    for theme, patterns in THEME_PATTERNS.items():
        scores[theme] = pattern_score(title_abstract, full_text, patterns)[0]
    return max(scores, key=scores.get), scores


def evidence_markdown(items: list[dict[str, object]]) -> list[str]:
    if not items:
        return ["- No reliable cue sentence was extracted; inspect the linked full text directly."]
    return [f"- [{item['section']}, p. {item['page']}] {item['text']}" for item in items]


def render_card(card: dict[str, object]) -> str:
    status = "WITHDRAWN - use only as a historical/collision record" if card["withdrawn"] else "active corpus record"
    lines = [
        f"# {card['paper_id']} - {card['title']}",
        "",
        f"- **Status:** {status}",
        f"- **Published:** {card['published']}",
        f"- **Authors:** {', '.join(card['authors'])}",
        f"- **Corpus class:** {card['directness']}",
        f"- **Primary theme:** {card['theme']}",
        f"- **Full text:** [{card['fulltext_kind']}]({card['fulltext_path']})",
        f"- **arXiv record:** {card['abs_url']}",
        "",
        "## Author Abstract",
        "",
        str(card["abstract"]),
        "",
        "## Traceable Contribution Evidence",
        "",
        *evidence_markdown(card["contribution_evidence"]),
        "",
        "## Traceable Result Evidence",
        "",
        *evidence_markdown(card["result_evidence"]),
        "",
        "## Limitations / Threats / Cautions",
        "",
        *evidence_markdown(card["limitation_evidence"]),
        "",
        "## Section Map",
        "",
    ]
    for name, section in card["sections"].items():
        lines.append(f"- `{name}`: starts near page {section['page']}; extracted {len(section['text'])} characters")
    if not card["sections"]:
        lines.append("- No conventional section headings were recovered; use full-text search.")
    lines.extend(["", "## Relevance to the Five Ideas", "", "| Idea | Score | Matched phrases |", "|---|---:|---|"])
    for idea, detail in card["idea_relevance"].items():
        phrases = ", ".join(detail["hits"].keys()) or "none"
        lines.append(f"| {idea} | {detail['score']} | {phrases} |")
    lines.extend([
        "",
        "## Reading Note",
        "",
        "This card is a machine-assisted evidence map over the complete local text, not a substitute for checking equations, tables, appendices, and cited baselines. The evidence snippets are locators for close reading, and withdrawn records must not be treated as validated findings.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--withdrawn", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    parser.add_argument("--cards-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--idea-audit", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    withdrawn_records = {
        item["paper_id"]: item for item in json.loads(args.withdrawn.read_text(encoding="utf-8"))
    }
    args.cards_dir.mkdir(parents=True, exist_ok=True)
    args.cards_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        paper_id = str(record["paper_id"])
        text_path = args.texts_dir / f"{paper_id}.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing full text for {paper_id}: {text_path}")
        full_text = clean_text(text_path.read_text(encoding="utf-8", errors="replace"))
        sections = section_map(full_text)
        lead = f"{record.get('title', '')}\n{record.get('abstract', '')}"
        idea_relevance: dict[str, dict[str, object]] = {}
        for idea, patterns in IDEA_PATTERNS.items():
            score, hits = pattern_score(lead, full_text, patterns)
            idea_relevance[idea] = {"score": score, "hits": hits}
        max_idea_score = max(int(item["score"]) for item in idea_relevance.values())
        theme, theme_scores = best_theme(lead, full_text)
        harness_hits = sum(full_text.lower().count(term) for term in ("harness", "scaffold", "orchestration"))
        if bool(record.get("core_selected")):
            directness = "core-selected"
        elif max_idea_score >= 24 or (harness_hits >= 12 and max_idea_score >= 10):
            directness = "direct-competitor"
        elif max_idea_score >= 10 or harness_hits >= 8:
            directness = "adjacent-method"
        else:
            directness = "peripheral-search-hit"

        withdrawn = withdrawn_records.get(paper_id)
        pdf_path = args.papers_dir / f"{paper_id}.pdf"
        if pdf_path.exists():
            fulltext_kind = "PDF"
            fulltext_path = f"../papers-all/{paper_id}.pdf"
        else:
            fulltext_kind = "archived HTML"
            fulltext_path = str(withdrawn["local_fulltext_path"]) if withdrawn else ""

        contribution = select_evidence(
            sections, ("abstract", "introduction", "method", "conclusion"), METHOD_CUES, 5
        )
        results = select_evidence(
            sections, ("results", "experiments", "conclusion", "abstract"), RESULT_CUES, 5
        )
        limitations = select_evidence(
            sections, ("limitations", "conclusion", "related_work"), None, 4
        )
        card: dict[str, object] = {
            "paper_id": paper_id,
            "title": record.get("title", ""),
            "abstract": record.get("abstract", ""),
            "authors": record.get("authors", []),
            "published": record.get("published", ""),
            "abs_url": record.get("abs_url", ""),
            "core_selected": bool(record.get("core_selected")),
            "all_rank": record.get("all_rank"),
            "withdrawn": bool(withdrawn),
            "withdrawal_reason": withdrawn.get("reason", "") if withdrawn else "",
            "fulltext_kind": fulltext_kind,
            "fulltext_path": fulltext_path,
            "text_path": str(text_path),
            "text_chars": len(full_text),
            "pages": full_text.count("\f") + 1,
            "sections": sections,
            "contribution_evidence": contribution,
            "result_evidence": results,
            "limitation_evidence": limitations,
            "idea_relevance": idea_relevance,
            "theme": theme,
            "theme_scores": theme_scores,
            "harness_hits": harness_hits,
            "directness": directness,
        }
        cards.append(card)
        (args.cards_dir / f"{paper_id}.md").write_text(render_card(card), encoding="utf-8")
        if index % 100 == 0 or index == len(records):
            print(f"[{index:04d}/{len(records):04d}] reading cards", flush=True)

    with args.cards_jsonl.open("w", encoding="utf-8") as handle:
        for card in cards:
            # ASCII escaping also protects JSONL framing from U+2028/U+2029 and
            # other Unicode characters that Python treats as line boundaries.
            handle.write(json.dumps(card, ensure_ascii=True) + "\n")

    directness_counts = Counter(str(card["directness"]) for card in cards)
    theme_counts = Counter(str(card["theme"]) for card in cards)
    summary = {
        "total": len(cards),
        "withdrawn": sum(bool(card["withdrawn"]) for card in cards),
        "directness": dict(directness_counts),
        "themes": dict(theme_counts),
        "text_chars": sum(int(card["text_chars"]) for card in cards),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    library_lines = [
        "# Full Corpus Library Index", "",
        f"Total records: **{len(cards)}**. Cards are evidence maps generated from local full text.", "",
        "| Rank | Paper | Year | Class | Theme | Full text | Reading card |", "|---:|---|---:|---|---|---|---|",
    ]
    for card in cards:
        year = str(card["published"])[:4]
        title = str(card["title"]).replace("|", "\\|")
        paper_id = str(card["paper_id"])
        rank = card["all_rank"] or "-"
        library_lines.append(
            f"| {rank} | {paper_id}: {title} | {year} | {card['directness']} | {card['theme']} | "
            f"[{card['fulltext_kind']}]({card['fulltext_path']}) | [card](../reading-cards/{paper_id}.md) |"
        )
    args.library_index.write_text("\n".join(library_lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# Full-Corpus Five-Idea Candidate Audit", "",
        "This is a ranking aid based on full-text phrase evidence. It is intentionally not a novelty verdict.", "",
    ]
    for idea in IDEA_PATTERNS:
        ranked = sorted(cards, key=lambda item: int(item["idea_relevance"][idea]["score"]), reverse=True)
        audit_lines.extend([f"## {idea}", "", "| Rank | Paper | Score | Class | Core |", "|---:|---|---:|---|---|"])
        for rank, card in enumerate(ranked[:40], start=1):
            paper_id = str(card["paper_id"])
            title = str(card["title"]).replace("|", "\\|")
            score = card["idea_relevance"][idea]["score"]
            audit_lines.append(
                f"| {rank} | [{paper_id}: {title}](../reading-cards/{paper_id}.md) | {score} | "
                f"{card['directness']} | {'yes' if card['core_selected'] else 'no'} |"
            )
        audit_lines.append("")
    args.idea_audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
