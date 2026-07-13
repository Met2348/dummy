#!/usr/bin/env python3
"""Verify the isolated TRACE-H literature and protocol workspace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["paper_id"]) for row in rows]
    errors: list[str] = []
    warnings: list[str] = []

    if len(ids) != len(set(ids)):
        errors.append("duplicate paper IDs in curated manifest")
    for paper_id in ids:
        pdf = args.root / "papers" / f"{paper_id}.pdf"
        text = args.root / "texts" / f"{paper_id}.txt"
        card = args.root / "evidence-cards" / f"{paper_id}.md"
        if not pdf.exists() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short PDF: {paper_id}")
        elif pdf.read_bytes()[:4] != b"%PDF":
            errors.append(f"invalid PDF header: {paper_id}")
        if not text.exists() or text.stat().st_size < 1_000:
            errors.append(f"missing or short text: {paper_id}")
        if not card.exists() or card.stat().st_size < 500:
            errors.append(f"missing or short evidence card: {paper_id}")

    schema_dir = args.root / "experiments" / "schemas"
    for schema in sorted(schema_dir.glob("*.json")):
        try:
            json.loads(schema.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON schema {schema.name}: {exc}")
    if len(list(schema_dir.glob("*.json"))) < 3:
        errors.append("expected three experiment schemas")

    all_markdown_files = list(args.root.rglob("*.md"))
    markdown_files = []
    excluded_markdown_files = []
    for path in all_markdown_files:
        relative_parts = path.relative_to(args.root).parts
        is_upstream_snapshot = "foundations" in relative_parts and "code" in relative_parts
        if "reading-cards" in relative_parts or is_upstream_snapshot:
            excluded_markdown_files.append(path)
        else:
            markdown_files.append(path)
    checked_links = 0
    for markdown in markdown_files:
        content = markdown.read_text(encoding="utf-8", errors="replace")
        for target in LINK_RE.findall(content):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:")) or target.startswith("#"):
                continue
            clean_target = target.split("#", 1)[0]
            if not clean_target:
                continue
            checked_links += 1
            resolved = (markdown.parent / clean_target).resolve()
            if not resolved.exists():
                warnings.append(f"broken relative link: {markdown.relative_to(args.root)} -> {target}")

    summary = {
        "curated_records": len(rows),
        "pdfs": len(list((args.root / "papers").glob("*.pdf"))),
        "texts": len(list((args.root / "texts").glob("*.txt"))),
        "evidence_cards": len([path for path in (args.root / "evidence-cards").glob("*.md") if path.name != "index.md"]),
        "schemas": len(list(schema_dir.glob("*.json"))),
        "markdown_files_checked": len(markdown_files),
        "markdown_files_excluded": len(excluded_markdown_files),
        "relative_links_checked": checked_links,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors and not warnings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors or warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
