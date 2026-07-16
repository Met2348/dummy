#!/usr/bin/env python3
"""Build a TRACE-H branch-race ALFWorld report from executed macro branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_branch_arg(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"invalid --branch value: {item!r}; expected NAME=path.json")
    name, path_text = item.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"invalid empty branch name in {item!r}")
    return name, Path(path_text)


def row_key(row: dict[str, Any], priority: int) -> tuple[int, float, int, int]:
    success = int(bool(row.get("success")))
    score = float(row.get("terminal_score", 0.0))
    steps = int(row.get("macro_steps", 0))
    return success, score, -steps, -priority


def branch_summary(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": name,
        "success": bool(row.get("success")),
        "terminal_score": float(row.get("terminal_score", 0.0)),
        "macro_steps": int(row.get("macro_steps", 0)),
        "search_order": row.get("search_order"),
        "policy_variant": row.get("policy_variant"),
        "horizon_switch_step": row.get("horizon_switch_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        action="append",
        required=True,
        help="Executed branch report as NAME=relative/or/absolute/path.json. Repeat for each branch.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("oracle", "first_success"),
        default="oracle",
        help="oracle selects the best branch after all branches are evaluated; first_success stops at the first successful branch.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    branch_items = [parse_branch_arg(item) for item in args.branch]
    reports = [(name, load_json(path), path) for name, path in branch_items]
    rows_by_branch = {
        name: {row["candidate_id"]: row for row in report["rows"]}
        for name, report, _path in reports
    }
    candidate_ids = [row["candidate_id"] for row in reports[0][1]["rows"]]
    expected = set(candidate_ids)
    for name, rows in rows_by_branch.items():
        found = set(rows)
        if found != expected:
            missing = sorted(expected - found)[:5]
            extra = sorted(found - expected)[:5]
            raise ValueError(
                f"branch {name!r} candidate set mismatch; missing={missing}, extra={extra}"
            )

    output_rows: list[dict[str, Any]] = []
    selection_counts = {name: 0 for name, _report, _path in reports}
    for candidate_id in candidate_ids:
        candidates = [
            (name, rows_by_branch[name][candidate_id], priority)
            for priority, (name, _report, _path) in enumerate(reports)
        ]
        if args.selection_mode == "first_success":
            evaluated = []
            selected_name = None
            selected_row = None
            for name, branch_row, priority in candidates:
                evaluated.append((name, branch_row, priority))
                if bool(branch_row.get("success")):
                    selected_name = name
                    selected_row = branch_row
                    break
            if selected_row is None or selected_name is None:
                selected_name, selected_row, _priority = max(
                    evaluated,
                    key=lambda item: row_key(item[1], item[2]),
                )
        else:
            evaluated = candidates
            selected_name, selected_row, _priority = max(
                candidates,
                key=lambda item: row_key(item[1], item[2]),
            )
        branch_summaries = [
            branch_summary(name, branch_row)
            for name, branch_row, _priority in evaluated
        ]
        selection_counts[selected_name] += 1
        row = dict(selected_row)
        row["selected_branch"] = selected_name
        row["branch_race_candidates"] = branch_summaries
        row["branch_eval_count"] = len(branch_summaries)
        row["branch_eval_steps_total"] = sum(item["macro_steps"] for item in branch_summaries)
        row["branch_eval_success_count"] = sum(bool(item["success"]) for item in branch_summaries)
        row["search_order"] = "traceh_branch_race"
        row["policy_variant"] = "branch_race_ledger"
        output_rows.append(row)

    result = {
        "experiment_id": "L3-ALFWORLD-BRANCH-RACE-REPORT",
        "selection_policy": [
            "Run each listed macro branch from the same replayable harness prefix.",
            (
                "Stop at the first successful branch in declared order."
                if args.selection_mode == "first_success"
                else "Choose a successful branch over any failed branch."
            ),
            (
                "If no branch succeeds, choose the best failed evaluated branch by score, steps, and declared order."
                if args.selection_mode == "first_success"
                else "Break success/score ties by fewer macro steps, then by the declared branch order."
            ),
        ],
        "selection_mode": args.selection_mode,
        "branch_reports": {
            name: str(path)
            for name, _report, path in reports
        },
        "branch_count": len(reports),
        "selection_counts": selection_counts,
        "candidate_count": len(output_rows),
        "success_count": sum(bool(row["success"]) for row in output_rows),
        "positive_score_count": sum(float(row["terminal_score"]) > 0.0 for row in output_rows),
        "rows": output_rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "branch_count": result["branch_count"],
                "candidate_count": result["candidate_count"],
                "success_count": result["success_count"],
                "positive_score_count": result["positive_score_count"],
                "selection_counts": result["selection_counts"],
                "ok": result["ok"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
