#!/usr/bin/env python3
"""Build a coverage-aware five-way PK summary from existing local reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


BLOCKS = [
    {
        "block": "qwen3_8b_gate",
        "candidates": "experiments/local-dev/reports/L3-source-policy-v2-qwen3-8b-no-progress-candidates.json",
        "source_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-8b-all-candidates-h80-v4-affordance.json",
        "target_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-8b-all-candidates-h80-v1.json",
    },
    {
        "block": "qwen3_4b_gate",
        "candidates": "experiments/local-dev/reports/L3-source-policy-v2-qwen3-4b-no-progress-candidates.json",
        "source_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-4b-gate-candidates-h80-v4-affordance.json",
        "target_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-4b-gate-candidates-h80-v1.json",
    },
    {
        "block": "qwen3_8b_expansion",
        "candidates": "experiments/local-dev/reports/L3-source-policy-v2-qwen3-8b-expansion-no-progress-candidates.json",
        "source_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-8b-expansion-candidates-h80-v4-affordance.json",
        "target_macro": "experiments/local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-8b-expansion-candidates-h80-v1.json",
    },
]

REPLAN_REPORTS = [
    "experiments/local-dev/reports/L3-source-policy-v2-qwen3-8b-none-replan-branch.json",
    "experiments/local-dev/reports/L3-source-policy-v2-qwen3-4b-p009-none-replan-branch.json",
    "experiments/local-dev/reports/L3-source-policy-v2-qwen3-8b-expansion-p036-none-replan-branch.json",
]

BUNDLE_REPORTS = [
    "experiments/local-dev/reports/L3-structured-action-smoke-qwen3-8b.json",
    "experiments/local-dev/reports/L3-structured-action-smoke-v2-qwen3-8b.json",
    "experiments/local-dev/reports/L3-structured-action-bundle-qwen3-4b-p009.json",
    "experiments/local-dev/reports/L3-structured-action-bundle-qwen3-8b-expansion-p036.json",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_from_trace(trace_path: str) -> dict[str, Any]:
    record = load_json(Path(trace_path))
    final = record["trace"][-1]
    score = float(final.get("score", 0.0))
    return {
        "score": score,
        "success": bool(score > 0.0),
        "steps": int(final.get("step_index", len(record["trace"]) - 1)) + 1,
        "trace_path": trace_path,
    }


def rows_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in report.get("rows", [])}


def load_replan_coverage(root: Path) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for relative in REPLAN_REPORTS:
        path = root / relative
        if not path.exists():
            continue
        report = load_json(path)
        for row in report.get("pairs", []):
            by_candidate.setdefault(str(row["candidate_id"]), []).append(
                {
                    "seed": int(row["seed"]),
                    "score": float(row["replan_score"]),
                    "success": bool(row["replan_success"]),
                    "report": relative,
                }
            )
    return {
        candidate_id: {
            "covered": True,
            "seed_count": len(rows),
            "best_score": max(row["score"] for row in rows),
            "mean_score": mean(row["score"] for row in rows),
            "success_any": any(row["success"] for row in rows),
            "success_count": sum(row["success"] for row in rows),
            "rows": rows,
        }
        for candidate_id, rows in by_candidate.items()
    }


def load_bundle_coverage(root: Path) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for relative in BUNDLE_REPORTS:
        path = root / relative
        if not path.exists():
            continue
        report = load_json(path)
        for row in report.get("rows", []):
            if row.get("action") != "BUNDLE_CONSERVATIVE":
                continue
            by_candidate.setdefault(str(row["candidate_id"]), []).append(
                {
                    "seed": int(row["seed"]),
                    "score": float(row["action_score"]),
                    "success": bool(row["success"]),
                    "selected_command": row.get("mechanism", {}).get("selected_command"),
                    "report": relative,
                }
            )
    return {
        candidate_id: {
            "covered": True,
            "run_count": len(rows),
            "best_score": max(row["score"] for row in rows),
            "mean_score": mean(row["score"] for row in rows),
            "success_any": any(row["success"] for row in rows),
            "success_count": sum(row["success"] for row in rows),
            "rows": rows,
        }
        for candidate_id, rows in by_candidate.items()
    }


def summarize_action(rows: list[dict[str, Any]], key: str, covered_key: str | None = None) -> dict[str, Any]:
    if covered_key:
        covered = [row for row in rows if row[covered_key]]
    else:
        covered = rows
    return {
        "covered_count": len(covered),
        "success_count": sum(bool(row[key]) for row in covered),
        "score_sum": sum(float(row[key.replace("_success", "_score")]) for row in covered)
        if key.endswith("_success")
        else None,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "block",
        "candidate_id",
        "goal",
        "prefix_step",
        "none_success",
        "target_first_success",
        "source_first_success",
        "natural_replan_covered",
        "natural_replan_success_any",
        "bundle_covered",
        "bundle_success_any",
        "source_minus_none",
        "source_minus_target",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    replan = load_replan_coverage(root)
    bundle = load_bundle_coverage(root)

    rows: list[dict[str, Any]] = []
    block_summaries: list[dict[str, Any]] = []
    for block_config in BLOCKS:
        candidate_report = load_json(root / block_config["candidates"])
        source_rows = rows_by_id(load_json(root / block_config["source_macro"]))
        target_rows = rows_by_id(load_json(root / block_config["target_macro"]))
        block_rows: list[dict[str, Any]] = []
        for candidate in candidate_report["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            source = source_rows[candidate_id]
            target = target_rows[candidate_id]
            none = terminal_from_trace(str(candidate["trace_path"]))
            replan_entry = replan.get(candidate_id)
            bundle_entry = bundle.get(candidate_id)
            row = {
                "block": block_config["block"],
                "candidate_id": candidate_id,
                "baseline_run_id": candidate["baseline_run_id"],
                "goal": source["goal"]["goal"],
                "prefix_step": int(candidate["step_index"]),
                "remaining_budget": 50 - int(candidate["step_index"]),
                "none_score": none["score"],
                "none_success": none["success"],
                "target_first_score": float(target["terminal_score"]),
                "target_first_success": bool(target["success"]),
                "target_first_steps": int(target["macro_steps"]),
                "source_first_score": float(source["terminal_score"]),
                "source_first_success": bool(source["success"]),
                "source_first_steps": int(source["macro_steps"]),
                "natural_replan_covered": bool(replan_entry),
                "natural_replan_best_score": replan_entry["best_score"] if replan_entry else None,
                "natural_replan_mean_score": replan_entry["mean_score"] if replan_entry else None,
                "natural_replan_success_any": replan_entry["success_any"] if replan_entry else None,
                "natural_replan_seed_count": replan_entry["seed_count"] if replan_entry else 0,
                "bundle_covered": bool(bundle_entry),
                "bundle_best_score": bundle_entry["best_score"] if bundle_entry else None,
                "bundle_mean_score": bundle_entry["mean_score"] if bundle_entry else None,
                "bundle_success_any": bundle_entry["success_any"] if bundle_entry else None,
                "bundle_run_count": bundle_entry["run_count"] if bundle_entry else 0,
                "source_minus_none": float(source["terminal_score"]) - none["score"],
                "target_minus_none": float(target["terminal_score"]) - none["score"],
                "source_minus_target": float(source["terminal_score"]) - float(target["terminal_score"]),
                "trace_path": none["trace_path"],
            }
            rows.append(row)
            block_rows.append(row)
        block_summaries.append(
            {
                "block": block_config["block"],
                "candidate_count": len(block_rows),
                "none_success": sum(row["none_success"] for row in block_rows),
                "target_first_success": sum(row["target_first_success"] for row in block_rows),
                "source_first_success": sum(row["source_first_success"] for row in block_rows),
                "source_only_vs_none": sum(
                    row["source_first_success"] and not row["none_success"]
                    for row in block_rows
                ),
                "none_only_vs_source": sum(
                    row["none_success"] and not row["source_first_success"]
                    for row in block_rows
                ),
                "natural_replan_covered": sum(row["natural_replan_covered"] for row in block_rows),
                "bundle_covered": sum(row["bundle_covered"] for row in block_rows),
            }
        )

    aggregate = {
        "candidate_count": len(rows),
        "none_success": sum(row["none_success"] for row in rows),
        "target_first_success": sum(row["target_first_success"] for row in rows),
        "source_first_success": sum(row["source_first_success"] for row in rows),
        "source_only_vs_none": sum(row["source_first_success"] and not row["none_success"] for row in rows),
        "none_only_vs_source": sum(row["none_success"] and not row["source_first_success"] for row in rows),
        "source_only_vs_target": sum(
            row["source_first_success"] and not row["target_first_success"] for row in rows
        ),
        "target_only_vs_source": sum(
            row["target_first_success"] and not row["source_first_success"] for row in rows
        ),
        "natural_replan_covered": sum(row["natural_replan_covered"] for row in rows),
        "natural_replan_success_any": sum(
            bool(row["natural_replan_success_any"]) for row in rows if row["natural_replan_covered"]
        ),
        "bundle_covered": sum(row["bundle_covered"] for row in rows),
        "bundle_success_any": sum(
            bool(row["bundle_success_any"]) for row in rows if row["bundle_covered"]
        ),
    }
    report = {
        "experiment_id": "L3-FIVE-WAY-PK-COVERAGE-AWARE-SUMMARY",
        "note": (
            "NONE, target-first macro, and source-first macro cover all rows. "
            "Natural REPLAN and bundle_conservative are report-level coverage from existing expensive branch runs only."
        ),
        "inputs": {
            "blocks": BLOCKS,
            "replan_reports": REPLAN_REPORTS,
            "bundle_reports": BUNDLE_REPORTS,
        },
        "aggregate": aggregate,
        "blocks": block_summaries,
        "rows": rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.tsv, rows)
    print(json.dumps({"aggregate": aggregate, "blocks": block_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
