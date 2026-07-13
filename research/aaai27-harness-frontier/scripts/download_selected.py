#!/usr/bin/env python3
"""Download PDFs listed in selected_papers.json."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.papers_dir.mkdir(parents=True, exist_ok=True)
    status: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        paper_id = str(row["paper_id"])
        url = str(row["pdf_url"])
        suffix = ".pdf"
        target = args.papers_dir / f"{paper_id.replace('/', '_')}{suffix}"
        state = "exists"
        error = ""
        try:
            if not target.exists() or target.stat().st_size < 10_000:
                request = urllib.request.Request(url, headers={"User-Agent": "aaai27-harness-frontier/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    target.write_bytes(response.read())
                state = "downloaded"
        except Exception as exc:  # noqa: BLE001 - keep batch running and log failure
            state = "failed"
            error = repr(exc)
        status.append(
            {
                "paper_id": paper_id,
                "state": state,
                "path": str(target),
                "bytes": target.stat().st_size if target.exists() else 0,
                "error": error,
            }
        )
        print(f"[{index:02d}/{len(rows)}] {state:10s} {paper_id}", flush=True)
        time.sleep(args.delay)

    status_path = args.manifest.parent / "download_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {status_path}", flush=True)


if __name__ == "__main__":
    main()

