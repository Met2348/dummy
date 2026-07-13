#!/usr/bin/env python3
"""Convert an archived paper HTML page into a section-preserving text file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    soup = BeautifulSoup(args.input.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for element in soup.select("script, style, nav, footer"):
        element.decompose()
    root = soup.select_one("article") or soup.body or soup

    lines: list[str] = []
    previous = ""
    for element in root.select("h1, h2, h3, h4, h5, h6, p, li, figcaption, pre"):
        value = clean(element.get_text(" ", strip=True))
        if not value or value == previous:
            continue
        if element.name and element.name.startswith("h"):
            level = min(int(element.name[1]), 6)
            value = f"{'#' * level} {value}"
        lines.append(value)
        previous = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"extracted {args.input.name} -> {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
