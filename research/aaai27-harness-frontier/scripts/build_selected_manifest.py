#!/usr/bin/env python3
"""Fetch canonical arXiv metadata for the hand-screened 70-paper core."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch(ids: list[str]) -> list[dict[str, object]]:
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    request = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "aaai27-harness-frontier/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        root = ET.fromstring(response.read())

    rows: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", NS):
        url = clean(entry.findtext("atom:id", namespaces=NS))
        paper_id = re.sub(r"v\d+$", "", url.rsplit("/", 1)[-1])
        primary = entry.find("arxiv:primary_category", NS)
        rows.append(
            {
                "paper_id": paper_id,
                "title": clean(entry.findtext("atom:title", namespaces=NS)),
                "abstract": clean(entry.findtext("atom:summary", namespaces=NS)),
                "authors": [
                    clean(author.findtext("atom:name", namespaces=NS))
                    for author in entry.findall("atom:author", NS)
                ],
                "published": clean(entry.findtext("atom:published", namespaces=NS)),
                "updated": clean(entry.findtext("atom:updated", namespaces=NS)),
                "primary_category": primary.attrib.get("term", "") if primary is not None else "",
                "abs_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--delay", type=float, default=3.2)
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8", newline="") as handle:
        seeds = list(csv.DictReader(handle, delimiter="\t"))
    seed_by_id = {row["paper_id"]: row for row in seeds}

    metadata: dict[str, dict[str, object]] = {}
    ids = [row["paper_id"] for row in seeds]
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start : start + args.batch_size]
        rows = fetch(batch)
        metadata.update({str(row["paper_id"]): row for row in rows})
        print(f"Fetched {len(rows):2d}/{len(batch):2d} records for batch {start // args.batch_size + 1}", flush=True)
        if start + args.batch_size < len(ids):
            time.sleep(args.delay)

    selected: list[dict[str, object]] = []
    missing: list[str] = []
    for rank, seed in enumerate(seeds, start=1):
        paper_id = seed["paper_id"]
        if paper_id not in metadata:
            missing.append(paper_id)
            continue
        row = dict(metadata[paper_id])
        row.update(
            {
                "selection_rank": rank,
                "category": seed["category"],
                "priority": seed["priority"],
            }
        )
        selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_rank",
                "paper_id",
                "published",
                "priority",
                "category",
                "title",
                "authors",
                "primary_category",
                "abs_url",
                "pdf_url",
            ],
        )
        writer.writeheader()
        for row in selected:
            flat = dict(row)
            flat["authors"] = "; ".join(row["authors"])
            flat.pop("abstract", None)
            flat.pop("updated", None)
            writer.writerow(flat)

    print(f"Wrote {len(selected)} selected papers; missing={missing}", flush=True)


if __name__ == "__main__":
    main()

