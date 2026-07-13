#!/usr/bin/env python3
"""Extract deterministic NO_PROGRESS branch candidates from baseline traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traceh_core.events import find_no_progress_prefixes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revisit-threshold", type=int, default=3)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--max-per-run", type=int, default=8)
    parser.add_argument("--min-step-gap", type=int, default=4)
    args = parser.parse_args()

    trace_paths = sorted(
        (args.trace_root / "records").glob(f"{args.run_prefix}-*.json")
    )
    if not trace_paths:
        raise FileNotFoundError(f"no traces for prefix {args.run_prefix!r}")

    candidates = []
    run_summaries = []
    for trace_path in trace_paths:
        record = json.loads(trace_path.read_text(encoding="utf-8"))
        detected = find_no_progress_prefixes(
            record["trace"],
            revisit_threshold=args.revisit_threshold,
            window=args.window,
        )
        selected = []
        for event in detected:
            if selected and event["step_index"] - selected[-1]["step_index"] < args.min_step_gap:
                continue
            selected.append(event)
            if len(selected) >= args.max_per_run:
                break
        for event in selected:
            candidates.append(
                {
                    "candidate_id": f"{record['run_id']}-p{event['step_index']:03d}",
                    "baseline_run_id": record["run_id"],
                    "task_id": record["task_id"],
                    "trace_path": str(trace_path),
                    **event,
                }
            )
        run_summaries.append(
            {
                "baseline_run_id": record["run_id"],
                "detected": len(detected),
                "selected": len(selected),
                "selected_steps": [item["step_index"] for item in selected],
            }
        )

    report = {
        "experiment_id": "L3-NO-PROGRESS-CANDIDATE-EXTRACTION",
        "source_run_prefix": args.run_prefix,
        "detector": {
            "event_type": "no_progress",
            "revisit_threshold": args.revisit_threshold,
            "window": args.window,
            "max_per_run": args.max_per_run,
            "min_step_gap": args.min_step_gap,
        },
        "run_summaries": run_summaries,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ok": bool(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
