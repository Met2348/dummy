#!/usr/bin/env python3
"""Freeze a priority queue for structured baseline follow-up runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Structured baseline priority queue",
        "",
        f"- Rule: {output['priority_rule']}",
        f"- Candidate count: {output['candidate_count']}",
        "",
        "| # | candidate | block | task family | prefix | trace |",
        "|---:|---|---|---|---:|---|",
    ]
    for index, item in enumerate(output["items"], 1):
        lines.append(
            "| {index} | `{candidate}` | {block} | `{family}` | {prefix} | `{trace}` |".format(
                index=index,
                candidate=item["candidate_id"],
                block=item["block"],
                family=item["task_family"],
                prefix=item["prefix_step"],
                trace=item.get("trace_path") or "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discordance", type=Path, required=True)
    parser.add_argument("--bucket", default="source_beats_both_none_and_target")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    report = load_json(args.discordance)
    if args.bucket not in report["buckets"]:
        raise ValueError(f"unknown bucket: {args.bucket}")
    items = list(report["buckets"][args.bucket])
    output = {
        "experiment_id": "L3-STRUCTURED-BASELINE-PRIORITY-QUEUE",
        "source_report": str(args.discordance),
        "priority_bucket": args.bucket,
        "priority_rule": "source_first_full succeeds while NONE and target_first_full both fail on the same dense prefix",
        "candidate_count": len(items),
        "candidate_ids": [item["candidate_id"] for item in items],
        "items": items,
        "ok": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": output["candidate_count"],
                "candidate_ids": output["candidate_ids"],
                "output": str(args.output),
                "markdown": str(args.markdown) if args.markdown else None,
                "ok": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
