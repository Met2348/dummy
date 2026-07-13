#!/usr/bin/env python3
"""Build a deduplicated arXiv candidate pool for harness research."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

QUERIES = [
    'all:"agent harness"',
    'all:"model harness"',
    'all:"harness engineering"',
    'all:"harness effect"',
    'all:"agent scaffold"',
    'all:"coding agent" AND all:scaffold',
    'all:"LLM agent" AND all:evaluation',
    'all:"LLM agents" AND all:benchmark',
    'all:"agentic system" AND all:evaluation',
    'all:"agent trajectory" AND all:evaluation',
    'all:"long-horizon agent" AND all:reliability',
    'all:"LLM agent" AND all:robustness',
    'all:"LLM agent" AND all:verification',
    'all:"LLM agent" AND all:"context management"',
    'all:"LLM agent" AND all:memory',
    'all:"LLM agent" AND all:"tool calling"',
    'all:"SWE-bench" AND all:agent',
    'all:"Terminal-Bench"',
    'all:"test-time scaling" AND all:agent',
    'all:"inference-time" AND all:agent',
    'all:causal AND all:"LLM agent" AND all:evaluation',
    'all:factorial AND all:agent AND all:evaluation',
    'all:"variance decomposition" AND all:agent',
    'all:"ranking stability" AND all:agent',
    'all:"benchmark overfitting" AND all:agent',
    'all:"system-level evaluation" AND all:agent',
]


def text_of(parent: ET.Element, path: str) -> str:
    node = parent.find(path, NS)
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()


def arxiv_id(url: str) -> str:
    value = url.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def score(record: dict[str, object]) -> tuple[int, list[str]]:
    title = str(record["title"]).lower()
    abstract = str(record["abstract"]).lower()
    blob = f"{title} {abstract}"
    points = 0
    reasons: list[str] = []

    rules = [
        (r"\b(harness|scaffold)\b", 12, 5, "explicit harness/scaffold"),
        (r"\b(system-level|agentic system|full stack)\b", 8, 4, "system-level"),
        (r"\b(evaluation|benchmark|leaderboard)\b", 5, 3, "evaluation"),
        (r"\b(causal|factorial|interaction effect|variance decomposition|attribution)\b", 9, 5, "attribution/statistics"),
        (r"\b(reliability|robustness|stability|failure)\b", 5, 3, "reliability"),
        (r"\b(trajectory|long-horizon|multi-turn)\b", 4, 2, "trajectory"),
        (r"\b(cost|budget|latency|efficiency|token)\b", 4, 2, "cost/budget"),
        (r"\b(context|memory|verification|retry|tool)\b", 3, 1, "harness component"),
        (r"\b(transfer|transport|generalization|distribution shift|overfit)\b", 6, 3, "transfer/overfit"),
    ]
    for pattern, title_score, abstract_score, reason in rules:
        if re.search(pattern, title):
            points += title_score
            reasons.append(reason)
        elif re.search(pattern, abstract):
            points += abstract_score
            reasons.append(reason)

    if re.search(r"\b(agent|agentic)\b", blob):
        points += 3
    year = int(str(record["published"])[:4])
    points += 4 if year >= 2026 else 2 if year >= 2025 else 0
    return points, sorted(set(reasons))


def fetch(query: str, max_results: int) -> list[dict[str, object]]:
    params = {
        "search_query": f"({query}) AND submittedDate:[202301010000 TO 202607102359]",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "aaai27-harness-frontier/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())

    rows: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", NS):
        abs_url = text_of(entry, "atom:id")
        aid = arxiv_id(abs_url)
        authors = [text_of(author, "atom:name") for author in entry.findall("atom:author", NS)]
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)]
        primary = entry.find("arxiv:primary_category", NS)
        rows.append(
            {
                "arxiv_id": aid,
                "title": text_of(entry, "atom:title"),
                "abstract": text_of(entry, "atom:summary"),
                "authors": authors,
                "published": text_of(entry, "atom:published"),
                "updated": text_of(entry, "atom:updated"),
                "categories": categories,
                "primary_category": primary.attrib.get("term", "") if primary is not None else "",
                "abs_url": f"https://arxiv.org/abs/{aid}",
                "pdf_url": f"https://arxiv.org/pdf/{aid}",
                "matched_queries": [query],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-results", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--query-start", type=int, default=1, help="One-based inclusive query index")
    parser.add_argument("--query-end", type=int, default=len(QUERIES), help="One-based inclusive query index")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_queries = QUERIES[args.query_start - 1 : args.query_end]

    merged: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    for local_index, query in enumerate(selected_queries, start=1):
        index = args.query_start + local_index - 1
        try:
            rows = fetch(query, args.max_results)
            for row in rows:
                aid = str(row["arxiv_id"])
                if aid in merged:
                    old_queries = list(merged[aid]["matched_queries"])
                    merged[aid]["matched_queries"] = sorted(set(old_queries + [query]))
                else:
                    merged[aid] = row
            print(f"[{index:02d}/{args.query_end:02d}] {len(rows):3d} rows | {len(merged):4d} unique | {query}", flush=True)
        except Exception as exc:  # noqa: BLE001 - preserve the search log
            failures.append({"query": query, "error": repr(exc)})
            print(f"[{index:02d}/{args.query_end:02d}] FAILED | {query} | {exc!r}", flush=True)
        time.sleep(args.delay)

    records = list(merged.values())
    for row in records:
        row["relevance_score"], row["score_reasons"] = score(row)
    records.sort(key=lambda row: (int(row["relevance_score"]), str(row["published"])), reverse=True)

    raw_path = args.output_dir / "arxiv_candidates.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = args.output_dir / "arxiv_candidates.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "arxiv_id",
                "published",
                "relevance_score",
                "title",
                "authors",
                "primary_category",
                "score_reasons",
                "matched_queries",
                "abs_url",
                "pdf_url",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "arxiv_id": row["arxiv_id"],
                    "published": row["published"],
                    "relevance_score": row["relevance_score"],
                    "title": row["title"],
                    "authors": "; ".join(row["authors"]),
                    "primary_category": row["primary_category"],
                    "score_reasons": "; ".join(row["score_reasons"]),
                    "matched_queries": " | ".join(row["matched_queries"]),
                    "abs_url": row["abs_url"],
                    "pdf_url": row["pdf_url"],
                }
            )

    log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": "2026-07-10T23:59:00+08:00",
        "queries": selected_queries,
        "query_count": len(selected_queries),
        "unique_records": len(records),
        "failures": failures,
    }
    (args.output_dir / "search_run.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} unique records to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
