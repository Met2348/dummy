#!/usr/bin/env python3
"""Resumable, validated bulk downloader for the complete arXiv candidate pool."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


USER_AGENT = "aaai27-harness-frontier/1.0 (academic literature archive)"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"%PDF"


class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                time.sleep(delay)
            self.next_time = time.monotonic() + self.interval


def seed_existing(paper_id: str, target: Path, seed_dir: Path | None) -> bool:
    if seed_dir is None:
        return False
    source = seed_dir / f"{paper_id.replace('/', '_')}.pdf"
    if not is_pdf(source):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return is_pdf(target)


def download_one(
    row: dict[str, object],
    papers_dir: Path,
    seed_dir: Path | None,
    limiter: RateLimiter,
    retries: int,
) -> dict[str, object]:
    paper_id = str(row["paper_id"])
    target = papers_dir / f"{paper_id.replace('/', '_')}.pdf"
    temp = target.with_suffix(".pdf.part")
    if is_pdf(target):
        return {"paper_id": paper_id, "state": "exists", "bytes": target.stat().st_size, "error": ""}
    if target.exists():
        target.unlink()
    if seed_existing(paper_id, target, seed_dir):
        return {"paper_id": paper_id, "state": "seeded", "bytes": target.stat().st_size, "error": ""}

    url = str(row["pdf_url"])
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            limiter.wait()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response, temp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if not is_pdf(temp):
                raise ValueError(f"invalid PDF payload ({temp.stat().st_size if temp.exists() else 0} bytes)")
            temp.replace(target)
            return {
                "paper_id": paper_id,
                "state": "downloaded",
                "bytes": target.stat().st_size,
                "attempts": attempt,
                "error": "",
            }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = repr(exc)
            if temp.exists():
                temp.unlink()
            if attempt < retries:
                time.sleep(min(60.0, 2.0**attempt))
    return {
        "paper_id": paper_id,
        "state": "failed",
        "bytes": target.stat().st_size if target.exists() else 0,
        "attempts": retries,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.4)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    args.papers_dir.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(args.request_interval)
    counts: dict[str, int] = {}
    completed = 0

    with args.status.open("w", encoding="utf-8") as status_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one, row, args.papers_dir, args.seed_dir, limiter, args.retries
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                state = str(result["state"])
                counts[state] = counts.get(state, 0) + 1
                status_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                status_handle.flush()
                print(
                    f"[{completed:04d}/{len(rows):04d}] {state:10s} {result['paper_id']} "
                    f"| downloaded={counts.get('downloaded', 0)} failed={counts.get('failed', 0)}",
                    flush=True,
                )

    summary = {"total": len(rows), "states": counts}
    args.status.with_name("download_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

