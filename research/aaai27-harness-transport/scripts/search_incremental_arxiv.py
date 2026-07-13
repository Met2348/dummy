#!/usr/bin/env python3
"""Run transport-specific arXiv searches and identify papers absent from the source corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
QUERIES = [
    'all:"cross-model transfer" AND all:agent',
    'all:"cross-model generalization" AND all:agent',
    'all:"held-out model" AND (all:harness OR all:scaffold)',
    'all:"unseen model" AND (all:harness OR all:scaffold)',
    'all:"model replacement" AND all:agent',
    'all:"model upgrade" AND all:harness',
    'all:"harness portability"',
    'all:"model-specific harness"',
    'all:"agent harness" AND all:transfer',
    'all:"agent scaffold" AND all:transfer',
    'all:"negative transfer" AND all:agent',
    'all:"ranking reversal" AND all:agent',
    'all:"test-time harness"',
    'all:"runtime harness adaptation"',
    'all:"model-scaffold" AND all:prediction',
    'all:"component interference" AND all:agent',
    'all:"model-harness interaction"',
    'all:transportability AND all:harness',
    'all:"cross-backend" AND all:agent',
]


def text_of(parent: ET.Element, path: str) -> str:
    node = parent.find(path, NS)
    return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""


def fetch(query: str, max_results: int, cutoff: str) -> list[dict[str, object]]:
    search_query = f"({query}) AND submittedDate:[202301010000 TO {cutoff}]"
    url = f"{API}?{urllib.parse.urlencode({'search_query': search_query, 'start': 0, 'max_results': max_results, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})}"
    request = urllib.request.Request(url, headers={"User-Agent": "aaai27-harness-transport/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        root = ET.fromstring(response.read())
    rows: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", NS):
        paper_id = re.sub(r"v\d+$", "", text_of(entry, "atom:id").rsplit("/", 1)[-1])
        primary = entry.find("arxiv:primary_category", NS)
        rows.append({
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": text_of(entry, "atom:title"),
            "abstract": text_of(entry, "atom:summary"),
            "authors": [text_of(author, "atom:name") for author in entry.findall("atom:author", NS)],
            "published": text_of(entry, "atom:published"),
            "updated": text_of(entry, "atom:updated"),
            "categories": [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)],
            "primary_category": primary.attrib.get("term", "") if primary is not None else "",
            "abs_url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
            "matched_queries": [query],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-results", type=int, default=80)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    source_ids = {
        str(json.loads(line)["paper_id"])
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    now = datetime.now(timezone.utc)
    cutoff = now.strftime("%Y%m%d%H%M")
    merged: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    query_counts: dict[str, int] = {}

    for index, query in enumerate(QUERIES, start=1):
        try:
            rows = fetch(query, args.max_results, cutoff)
            query_counts[query] = len(rows)
            for row in rows:
                paper_id = str(row["paper_id"])
                if paper_id in merged:
                    merged[paper_id]["matched_queries"] = sorted(set(merged[paper_id]["matched_queries"] + [query]))
                else:
                    merged[paper_id] = row
            print(f"[{index:02d}/{len(QUERIES):02d}] {len(rows):3d} rows | {len(merged):4d} unique | {query}", flush=True)
        except Exception as exc:  # Preserve failed queries in the audit log.
            failures.append({"query": query, "error": repr(exc)})
            print(f"[{index:02d}/{len(QUERIES):02d}] FAILED | {query} | {exc!r}", flush=True)
        if index < len(QUERIES):
            time.sleep(args.delay)

    rows = sorted(merged.values(), key=lambda row: (len(row["matched_queries"]), row["published"]), reverse=True)
    for row in rows:
        row["already_in_source_corpus"] = str(row["paper_id"]) in source_ids
        row["incremental_query_count"] = len(row["matched_queries"])
    new_rows = [row for row in rows if not row["already_in_source_corpus"]]

    for path, values in ((args.output, rows), (args.new_output, new_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    log = {
        "generated_at": now.isoformat(),
        "cutoff_utc": cutoff,
        "source_corpus_count": len(source_ids),
        "query_count": len(QUERIES),
        "unique_hits": len(rows),
        "already_in_source_corpus": len(rows) - len(new_rows),
        "new_hits": len(new_rows),
        "query_counts": query_counts,
        "failures": failures,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
