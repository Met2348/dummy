#!/usr/bin/env python3
"""Fetch arXiv metadata for ranked citation-gap IDs."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path


API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def text_of(parent: ET.Element, path: str) -> str:
    node = parent.find(path, NS)
    return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""


def fetch(ids: list[str]) -> list[dict[str, object]]:
    url = f"{API}?{urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})}"
    request = urllib.request.Request(url, headers={"User-Agent": "aaai27-harness-frontier/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        root = ET.fromstring(response.read())
    rows: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", NS):
        abs_url = text_of(entry, "atom:id")
        paper_id = re.sub(r"v\d+$", "", abs_url.rsplit("/", 1)[-1])
        primary = entry.find("arxiv:primary_category", NS)
        rows.append(
            {
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
            }
        )
    return rows


def fetch_with_retry(ids: list[str], retries: int) -> list[dict[str, object]]:
    for attempt in range(1, retries + 1):
        try:
            return fetch(ids)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries:
                raise
            wait_seconds = 15 * attempt
            print(f"rate limited; waiting {wait_seconds}s (attempt {attempt}/{retries})", flush=True)
            time.sleep(wait_seconds)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            wait_seconds = 5 * attempt
            print(f"network retry in {wait_seconds}s (attempt {attempt}/{retries})", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--year-prefix", default="", help="Optional arXiv ID prefix such as 26")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    gaps = [json.loads(line) for line in args.gaps.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.year_prefix:
        gaps = [row for row in gaps if str(row["paper_id"]).startswith(args.year_prefix)]
    gaps = gaps[: args.limit]
    gap_by_id = {str(row["paper_id"]): row for row in gaps}
    records: list[dict[str, object]] = []
    ids = list(gap_by_id)
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start : start + args.batch_size]
        fetched = fetch_with_retry(batch, args.retries)
        for row in fetched:
            gap = gap_by_id.get(str(row["paper_id"]), {})
            row.update({key: value for key, value in gap.items() if key != "contexts"})
            row["citation_contexts"] = gap.get("contexts", [])
            records.append(row)
        print(f"[{min(start + len(batch), len(ids)):04d}/{len(ids):04d}] metadata", flush=True)
        if start + args.batch_size < len(ids):
            time.sleep(args.delay)
    records.sort(key=lambda row: int(row.get("citation_gap_score", 0)), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"wrote {len(records)} metadata records")


if __name__ == "__main__":
    main()
