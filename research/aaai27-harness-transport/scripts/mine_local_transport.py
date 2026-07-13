#!/usr/bin/env python3
"""Mine model-to-model harness transport evidence from the local full-text corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# These phrases assert a transport/generalization claim. Generic occurrences of
# "source model" and "target model" are only supporting evidence.
TRANSPORT_PHRASES: dict[str, int] = {
    "transportability": 24,
    "harness transfer": 28,
    "cross-model transfer": 28,
    "cross model transfer": 28,
    "cross-model gain": 24,
    "cross model gain": 24,
    "cross-model generalization": 24,
    "cross model generalization": 24,
    "cross-model portability": 28,
    "cross model portability": 28,
    "cross-backend": 22,
    "backend transfer": 22,
    "held-out model": 24,
    "held out model": 24,
    "unseen model": 22,
    "model upgrade": 26,
    "model replacement": 26,
    "model substitution": 26,
    "model-specific harness": 28,
    "model specific harness": 28,
    "model-harness interaction": 26,
    "model harness interaction": 26,
    "ranking reversal": 24,
    "rank reversal": 24,
    "negative transfer": 22,
    "cross-llm": 20,
    "cross llm": 20,
    "portable harness": 28,
    "harness portability": 28,
    "backbone transfer": 20,
    "transfer to other models": 24,
    "transfer to 17 other models": 28,
    "alternate base models": 20,
    "additional model backbones": 18,
}

SUPPORT_PHRASES: dict[str, int] = {
    "source model": 4,
    "target model": 4,
    "source backbone": 4,
    "target backbone": 4,
    "warm start": 5,
    "warm-start": 5,
    "without retraining": 4,
    "without target-specific": 6,
}

HARNESS_TERMS = (
    "harness", "scaffold", "orchestration", "runtime", "agent interface",
    "prompt", "skill", "memory", "tool", "verification", "retry", "context",
    "guidance", "controller", "contract", "playbook", "repair layer",
)

PREDICTION_TERMS = (
    "predict", "prediction", "forecast", "unseen combination", "held-out cell",
    "sealed target", "out-of-sample", "leave-one-model-out", "prospective",
)

ATOMIC_TERMS = (
    "ablation", "component", "factorial", "intervention", "treatment effect",
    "marginal effect", "leave-one-layer-out", "one factor", "controlled variant",
)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def windows(text: str, phrases: list[str], radius: int = 650) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for phrase in phrases:
        start = 0
        while len(found) < 80:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            found.append(lowered[max(0, index - radius): index + len(phrase) + radius])
            start = index + len(phrase)
    return found


def snippets(text: str, phrase_hits: list[str], limit: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lowered = text.lower()
    seen: set[tuple[str, int]] = set()
    for phrase in phrase_hits:
        start = 0
        while len(rows) < limit:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            page = text.count("\f", 0, index) + 1
            key = (phrase, page)
            if key not in seen:
                left = max(0, index - 300)
                right = min(len(text), index + len(phrase) + 500)
                rows.append({"phrase": phrase, "page": page, "snippet": compact(text[left:right])})
                seen.add(key)
            start = index + len(phrase)
    return rows


def classify(
    lead: str,
    local_context: str,
    transport_hits: dict[str, int],
    prediction_hits: int,
    atomic_hits: int,
) -> tuple[str, str]:
    lead_lower = lead.lower()
    has_harness_object = any(term in local_context for term in HARNESS_TERMS)
    harness_in_lead = any(term in lead_lower for term in HARNESS_TERMS)
    has_explicit_harness = "harness" in local_context or "scaffold" in local_context
    has_model_holdout = any(
        phrase in transport_hits
        for phrase in ("held-out model", "held out model", "unseen model", "model replacement")
    )
    if has_explicit_harness and transport_hits:
        tier = "A-direct-harness-transport"
    elif has_harness_object and transport_hits:
        tier = "B-component-transport"
    else:
        tier = "C-adjacent-method"

    if prediction_hits and has_model_holdout and (harness_in_lead or has_harness_object):
        role = "predictive-collision"
    elif atomic_hits and has_harness_object:
        role = "atomicity-or-interaction"
    elif "negative transfer" in transport_hits or "ranking reversal" in transport_hits or "rank reversal" in transport_hits:
        role = "counterevidence"
    elif tier.startswith("A-"):
        role = "direct-transfer"
    elif tier.startswith("B-"):
        role = "component-transfer"
    else:
        role = "methodological-background"
    return tier, role


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-score", type=int, default=24)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    cards = {
        str(card["paper_id"]): card
        for card in (json.loads(line) for line in args.cards.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    ranked: list[dict[str, object]] = []

    for record in records:
        paper_id = str(record["paper_id"])
        text_path = args.texts_dir / f"{paper_id}.txt"
        if not text_path.exists():
            continue
        full_text = text_path.read_text(encoding="utf-8", errors="replace").replace("\x00", " ")
        full_lower = full_text.lower()
        lead = f"{record.get('title', '')}\n{record.get('abstract', '')}"
        lead_lower = lead.lower()

        transport_hits = {phrase: full_lower.count(phrase) for phrase in TRANSPORT_PHRASES if phrase in full_lower}
        if not transport_hits:
            continue
        context_windows = windows(full_text, list(transport_hits))
        local_context = "\n".join(context_windows)
        local_harness_terms = {term: local_context.count(term) for term in HARNESS_TERMS if term in local_context}
        if not local_harness_terms:
            continue

        support_hits = {phrase: local_context.count(phrase) for phrase in SUPPORT_PHRASES if phrase in local_context}
        prediction_hits = sum(local_context.count(term) for term in PREDICTION_TERMS)
        atomic_hits = sum(local_context.count(term) for term in ATOMIC_TERMS)
        score = 0
        for phrase, count in transport_hits.items():
            score += min(count, 4) * TRANSPORT_PHRASES[phrase]
            score += lead_lower.count(phrase) * TRANSPORT_PHRASES[phrase] * 3
        score += sum(min(count, 4) * SUPPORT_PHRASES[phrase] for phrase, count in support_hits.items())
        score += min(len(local_harness_terms), 8) * 5
        score += min(prediction_hits, 8) * 4
        score += min(atomic_hits, 8) * 3
        if "harness" in lead_lower or "scaffold" in lead_lower:
            score += 18
        if record.get("core_selected"):
            score += 5
        if score < args.minimum_score:
            continue

        tier, role = classify(lead, local_context, transport_hits, prediction_hits, atomic_hits)
        card = cards.get(paper_id, {})
        row = dict(record)
        row.update({
            "transport_score": score,
            "transport_tier": tier,
            "transport_role": role,
            "transport_phrase_hits": transport_hits,
            "support_phrase_hits": support_hits,
            "local_harness_terms": local_harness_terms,
            "prediction_term_hits": prediction_hits,
            "atomic_term_hits": atomic_hits,
            "transport_snippets": snippets(full_text, list(transport_hits)),
            "source_text_path": str(text_path),
            "source_card_path": f"research/aaai27-harness-frontier/reading-cards/{paper_id}.md",
            "source_directness": card.get("directness", ""),
        })
        ranked.append(row)

    ranked.sort(key=lambda row: (int(row["transport_score"]), str(row.get("published", ""))), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["transport_rank"] = index
        row["deep_read"] = row["transport_tier"].startswith("A-") or row["transport_role"] in {
            "predictive-collision", "counterevidence", "atomicity-or-interaction"
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    tier_counts: dict[str, int] = {}
    for row in ranked:
        tier = str(row["transport_tier"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    lines = [
        "# Local Full-Text Transport Search", "",
        f"Candidates with explicit transport evidence: **{len(ranked)}**.",
        "This is a screening index. Tier A still requires manual verification of the transferred object and target-model protocol.", "",
        "## Tier counts", "",
    ]
    lines.extend(f"- `{tier}`: {count}" for tier, count in sorted(tier_counts.items()))
    lines.extend(["", "| Rank | Paper | Score | Tier | Role | Prediction | Atomic | Evidence phrases |", "|---:|---|---:|---|---|---:|---:|---|"])
    for row in ranked:
        paper_id = str(row["paper_id"])
        title = str(row["title"]).replace("|", "\\|")
        phrases = ", ".join(row["transport_phrase_hits"].keys()).replace("|", "\\|")
        lines.append(
            f"| {row['transport_rank']} | [{paper_id}: {title}](../../aaai27-harness-frontier/reading-cards/{paper_id}.md) | "
            f"{row['transport_score']} | {row['transport_tier']} | {row['transport_role']} | "
            f"{row['prediction_term_hits']} | {row['atomic_term_hits']} | {phrases} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ranked {len(ranked)} transport candidates: {tier_counts}")


if __name__ == "__main__":
    main()
