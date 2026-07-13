#!/usr/bin/env python3
"""Build a reproducible cross-model audit from source branch reports and traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from traceh_core.source_pilot_analysis import (
    combine_source_pilots,
    summarize_model_source_pilot,
)


def load_trace_records(records_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record["run_id"])] = record
    return records


def find_record(
    records: dict[str, dict[str, Any]],
    *,
    candidate_id: str,
    action: str,
    seed: int,
) -> dict[str, Any]:
    suffix = f"-{candidate_id}-{action.lower()}-s{seed}"
    matches = [record for run_id, record in records.items() if run_id.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one trace ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def pair_traces(
    report: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    paired = []
    for pair in report["pairs"]:
        candidate_id = str(pair["candidate_id"])
        seed = int(pair["seed"])
        none = find_record(
            records,
            candidate_id=candidate_id,
            action="NONE",
            seed=seed,
        )
        replan = find_record(
            records,
            candidate_id=candidate_id,
            action="REPLAN",
            seed=seed,
        )
        paired.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "none": none["trace"],
                "replan": replan["trace"],
                "plan": replan["plan"],
                "none_extra_model_calls": none["extra_model_calls"],
                "replan_extra_model_calls": replan["extra_model_calls"],
                "parser_failure": bool(none["parser_failure"] or replan["parser_failure"]),
            }
        )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("REPORT", "TRACE_RECORDS_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    source_inputs = []
    for report_text, trace_dir_text in args.model:
        report_path = Path(report_text)
        trace_dir = Path(trace_dir_text)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = load_trace_records(trace_dir)
        summary = summarize_model_source_pilot(
            report,
            pair_traces(report, records),
        )
        summary["historical_report_decision"] = report["decision"]["decision"]
        summaries.append(summary)
        source_inputs.append(
            {
                "report": str(report_path),
                "trace_records_dir": str(trace_dir),
                "trace_record_count": len(records),
            }
        )

    combined = combine_source_pilots(summaries)
    output = {
        "experiment_id": "L3-MULTIMODEL-SOURCE-PILOT-AUDIT",
        "source_inputs": source_inputs,
        "models": summaries,
        "combined": combined,
        "decision_override": {
            "supersedes_historical_8b_label": "MOVE_TO_QWEN3_8B",
            "reason": "The 8B run is already the escalation target; another identical label is stale.",
            "effective_decision": combined["decision"],
        },
        "evidence_boundary": {
            "source_branch_policy_tested": True,
            "response_bank_formed": False,
            "transport_algorithm_tested": False,
            "target_executor_queried": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
