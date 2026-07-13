from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    checks = []
    checks.append(("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(("pymupdf", has_module("fitz"), "PDF extraction"))
    checks.append(("beautifulsoup4", has_module("bs4"), "HTML extraction"))

    manifest = ROOT / "metadata" / "papers_manifest.json"
    papers = ROOT / "papers"
    reports = ROOT / "reports"
    corpus = ROOT / "metadata" / "corpus.jsonl"

    manifest_items = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    checks.append(("manifest_count_70", len(manifest_items) == 70, str(len(manifest_items))))
    checks.append(("paper_files_70", papers.exists() and len(list(papers.iterdir())) == 70, str(len(list(papers.iterdir()))) if papers.exists() else "0"))
    checks.append(("reports_70", reports.exists() and len(list(reports.glob("*group-meeting-report.md"))) == 70, str(len(list(reports.glob("*group-meeting-report.md")))) if reports.exists() else "0"))
    checks.append(("corpus_exists", corpus.exists(), str(corpus)))

    ok = True
    for name, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        print(f"{status:4} {name:20} {detail}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
