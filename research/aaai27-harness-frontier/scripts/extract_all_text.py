#!/usr/bin/env python3
"""Extract text from every validated PDF with resumable status output."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract(pdf: Path, text_dir: Path, pdftotext: Path) -> dict[str, object]:
    target = text_dir / f"{pdf.stem}.txt"
    if target.exists() and target.stat().st_size >= 1_000:
        return {"paper_id": pdf.stem, "state": "exists", "chars": target.stat().st_size, "error": ""}
    process = subprocess.run(
        [str(pdftotext), "-layout", str(pdf), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    size = target.stat().st_size if target.exists() else 0
    state = "extracted" if process.returncode == 0 and size >= 1_000 else "failed"
    return {
        "paper_id": pdf.stem,
        "state": state,
        "chars": size,
        "error": process.stderr[-1000:] if state == "failed" else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--pdftotext", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pdfs = sorted(args.papers_dir.glob("*.pdf"))
    args.text_dir.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.status.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract, pdf, args.text_dir, args.pdftotext): pdf for pdf in pdfs}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                state = str(result["state"])
                counts[state] = counts.get(state, 0) + 1
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index:04d}/{len(pdfs):04d}] {state:10s} {result['paper_id']} "
                    f"| failed={counts.get('failed', 0)}",
                    flush=True,
                )
    summary = {"total": len(pdfs), "states": counts}
    args.status.with_name("extract_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

