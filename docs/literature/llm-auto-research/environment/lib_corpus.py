from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz

        doc = fitz.open(path)
        pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
        return normalize_text("\n".join(pages))
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return normalize_text(soup.get_text(" ", strip=True))
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def chunk_text(text: str, max_chars: int = 2200, overlap: int = 200) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split = text.rfind(" ", start + max_chars // 2, end)
            if split > start:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_manifest(root: Path) -> list[dict]:
    manifest_path = root / "metadata" / "papers_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_records(root: Path, max_chars: int = 2200, overlap: int = 200) -> list[dict]:
    records: list[dict] = []
    for item in load_manifest(root):
        paper_path = root / "papers" / item["filename"]
        text = extract_text_from_file(paper_path)
        for chunk_id, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
            records.append(
                {
                    "paper_id": item["id"],
                    "key": item["key"],
                    "title": item["title"],
                    "year": item.get("year", ""),
                    "category": item.get("category", ""),
                    "source_url": item.get("source_url", ""),
                    "paper_path": str(paper_path).replace("\\", "/"),
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
