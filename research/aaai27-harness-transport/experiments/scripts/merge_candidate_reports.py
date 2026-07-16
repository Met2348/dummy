#!/usr/bin/env python3
"""Merge multiple no-progress candidate reports into one block-aware report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--block", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.input) != len(args.block):
        raise ValueError("--input and --block must have the same count")

    candidates: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path, block in zip(args.input, args.block, strict=True):
        report = load_json(path)
        block_candidates = []
        for item in report["candidates"]:
            candidate = dict(item)
            candidate["block"] = block
            candidate["source_report"] = str(path)
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            seen_ids.add(candidate_id)
            candidates.append(candidate)
            block_candidates.append(candidate)
        blocks.append(
            {
                "block": block,
                "source_report": str(path),
                "candidate_count": len(block_candidates),
                "run_count": len(report.get("run_summaries", [])),
            }
        )

    output = {
        "experiment_id": "L3-MERGED-NO-PROGRESS-CANDIDATES",
        "input_reports": [str(path) for path in args.input],
        "blocks": blocks,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ok": bool(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
