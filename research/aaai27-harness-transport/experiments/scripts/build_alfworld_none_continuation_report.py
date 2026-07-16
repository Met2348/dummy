#!/usr/bin/env python3
"""Convert candidate source continuations into a macro-report-shaped branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = load_json(args.candidates)
    rows: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        trace_record = load_json(Path(candidate["trace_path"]))
        trace = trace_record["trace"]
        step_index = int(candidate["step_index"])
        final = trace[-1]
        score = float(final.get("score", 0.0))
        suffix = trace[step_index:]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "baseline_run_id": candidate["baseline_run_id"],
                "task_id": candidate["task_id"],
                "prefix_state_hash": candidate["prefix_state_hash"],
                "replayed_state_hash": candidate["prefix_state_hash"],
                "goal": {},
                "max_macro_steps": len(suffix),
                "search_order": "none_continuation",
                "policy_variant": "defer_to_source_policy",
                "horizon_switch_step": None,
                "macro_steps": len(suffix),
                "terminal_score": score,
                "success": bool(final.get("success", score > 0.0)),
                "visited_count": None,
                "delivered_objects": [],
                "transformed_objects": [],
                "deposit_target": None,
                "trace": suffix,
            }
        )

    result = {
        "experiment_id": "L3-ALFWORLD-NONE-CONTINUATION-REPORT",
        "candidate_report": str(args.candidates),
        "candidate_count": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "positive_score_count": sum(float(row["terminal_score"]) > 0.0 for row in rows),
        "rows": rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "candidate_count": result["candidate_count"],
                "success_count": result["success_count"],
                "positive_score_count": result["positive_score_count"],
                "ok": result["ok"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
