#!/usr/bin/env python3
"""Aggregate dense local macro PK reports into summary and matrix files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHOD_REPORTS = {
    "target_first_full": "experiments/local-dev/reports/L3-dense-macro-target-first-full-h80-20260714.json",
    "source_first_full": "experiments/local-dev/reports/L3-dense-macro-source-first-full-h80-20260714.json",
    "source_first_no_history_ledger": "experiments/local-dev/reports/L3-dense-macro-source-first-no-history-ledger-h80-20260714.json",
    "source_first_no_deposit_lock": "experiments/local-dev/reports/L3-dense-macro-source-first-no-deposit-lock-h80-20260714.json",
    "source_first_no_inventory_inference": "experiments/local-dev/reports/L3-dense-macro-source-first-no-inventory-inference-h80-20260714.json",
    "source_first_no_instance_ledger": "experiments/local-dev/reports/L3-dense-macro-source-first-no-instance-ledger-h80-20260714.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_from_trace(trace_path: str) -> dict[str, Any]:
    record = load_json(Path(trace_path))
    final = record["trace"][-1]
    score = float(final.get("score", 0.0))
    return {
        "score": score,
        "success": bool(final.get("success", score > 0.0)),
        "steps": len(record["trace"]),
    }


def load_method_rows(root: Path, relative: str) -> dict[str, dict[str, Any]]:
    report = load_json(root / relative)
    return {row["candidate_id"]: row for row in report["rows"]}


def summarize_rows(
    rows: list[dict[str, Any]],
    method_names: list[str],
    *,
    source_method: str,
    target_method: str,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "candidate_count": len(rows),
        "none_success": sum(row["none_success"] for row in rows),
    }
    for method in method_names:
        aggregate[f"{method}_success"] = sum(row[f"{method}_success"] for row in rows)
        aggregate[f"{method}_mean_steps"] = (
            sum(row[f"{method}_macro_steps"] for row in rows) / len(rows) if rows else 0.0
        )
    source = source_method
    target = target_method
    aggregate["source_only_vs_none"] = sum(
        row[f"{source}_success"] and not row["none_success"] for row in rows
    )
    aggregate["none_only_vs_source"] = sum(
        row["none_success"] and not row[f"{source}_success"] for row in rows
    )
    aggregate["source_only_vs_target"] = sum(
        row[f"{source}_success"] and not row[f"{target}_success"] for row in rows
    )
    aggregate["target_only_vs_source"] = sum(
        row[f"{target}_success"] and not row[f"{source}_success"] for row in rows
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("experiments/local-dev/reports/L3-source-policy-v2-dense-merged-candidates-20260714.json"),
    )
    parser.add_argument(
        "--method-report",
        action="append",
        help="Override method reports as METHOD=relative/path.json. May be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--source-method", default="source_first_full")
    parser.add_argument("--target-method", default="target_first_full")
    args = parser.parse_args()

    root = args.root
    candidates_report = load_json(root / args.candidates)
    candidates = candidates_report["candidates"]
    method_reports = dict(METHOD_REPORTS)
    if args.method_report:
        method_reports = {}
        for item in args.method_report:
            if "=" not in item:
                raise ValueError(f"invalid --method-report value: {item!r}")
            name, relative = item.split("=", 1)
            method_reports[name] = relative
    method_names = list(method_reports)
    if args.source_method not in method_names:
        raise ValueError(f"--source-method is not in method reports: {args.source_method}")
    if args.target_method not in method_names:
        raise ValueError(f"--target-method is not in method reports: {args.target_method}")
    method_rows = {
        name: load_method_rows(root, relative)
        for name, relative in method_reports.items()
    }

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        terminal = terminal_from_trace(candidate["trace_path"])
        row: dict[str, Any] = {
            "block": candidate.get("block", "unknown"),
            "candidate_id": candidate_id,
            "baseline_run_id": candidate["baseline_run_id"],
            "task_id": candidate["task_id"],
            "prefix_step": int(candidate["step_index"]),
            "none_score": terminal["score"],
            "none_success": terminal["success"],
            "none_steps": terminal["steps"],
        }
        for method in method_names:
            method_row = method_rows[method][candidate_id]
            row[f"{method}_score"] = float(method_row["terminal_score"])
            row[f"{method}_success"] = bool(method_row["success"])
            row[f"{method}_macro_steps"] = int(method_row["macro_steps"])
            row[f"{method}_branch_eval_count"] = int(method_row.get("branch_eval_count", 1))
            row[f"{method}_branch_eval_steps_total"] = int(
                method_row.get("branch_eval_steps_total", method_row["macro_steps"])
            )
            row[f"{method}_branch_eval_success_count"] = int(
                method_row.get("branch_eval_success_count", int(bool(method_row["success"])))
            )
            row[f"{method}_delivered_count"] = len(method_row.get("delivered_objects", []))
            row[f"{method}_deposit_target"] = method_row.get("deposit_target")
        rows.append(row)

    block_summaries = []
    for block in sorted({row["block"] for row in rows}):
        block_rows = [row for row in rows if row["block"] == block]
        block_summaries.append(
            {
                "block": block,
                **summarize_rows(
                    block_rows,
                    method_names,
                    source_method=args.source_method,
                    target_method=args.target_method,
                ),
            }
        )

    output = {
        "experiment_id": "L3-DENSE-MACRO-PK-SUMMARY",
        "candidate_report": str(args.candidates),
        "method_reports": method_reports,
        "source_method": args.source_method,
        "target_method": args.target_method,
        "aggregate": summarize_rows(
            rows,
            method_names,
            source_method=args.source_method,
            target_method=args.target_method,
        ),
        "blocks": block_summaries,
        "rows": rows,
        "notes": [
            "This is development evidence over dense no-progress prefixes, not a sealed task-level final result.",
            "All macro methods are no-LLM replay interventions over existing source-policy traces.",
        ],
        "ok": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "block",
        "candidate_id",
        "prefix_step",
        "none_success",
        *[f"{method}_success" for method in method_names],
        "source_only_vs_none",
        "source_only_vs_target",
    ]
    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            source_success = row[f"{args.source_method}_success"]
            writer.writerow(
                {
                    **{name: row[name] for name in fieldnames if name in row},
                    "source_only_vs_none": source_success and not row["none_success"],
                    "source_only_vs_target": source_success and not row[f"{args.target_method}_success"],
                }
            )

    print(json.dumps({"aggregate": output["aggregate"], "blocks": block_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
