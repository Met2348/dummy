#!/usr/bin/env python3
"""Build a compute-cost audit for ALFWorld macro and branch-race reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


METHOD_SPECS = [
    ("none", "NONE / source policy", "baseline"),
    ("alphabetical_full", "alphabetical macro", "baseline"),
    ("reverse_container_full", "reverse-container macro", "baseline"),
    ("baseline_branch_race_with_defer", "baseline-only branch race", "baseline"),
    ("source_exhaustive_full", "source-prioritized exhaustive sweep", "ablation"),
    ("source_horizon_aware_full", "TRACE-H branch-race ledger", "ours"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sign_test(diffs: list[float], eps: float = 1e-12) -> dict[str, Any]:
    wins = sum(diff > eps for diff in diffs)
    losses = sum(diff < -eps for diff in diffs)
    ties = len(diffs) - wins - losses
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    else:
        smaller = min(wins, losses)
        p_value = 2.0 * sum(math.comb(trials, index) for index in range(smaller + 1)) / (2**trials)
        p_value = min(1.0, p_value)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "trials": trials,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def method_values(row: dict[str, Any], method: str) -> dict[str, float]:
    if method == "none":
        success = float(row["none_success"])
        selected_steps = float(row["none_steps"])
        eval_steps = selected_steps
        eval_count = 1.0
    else:
        success = float(row[f"{method}_success"])
        selected_steps = float(row[f"{method}_macro_steps"])
        eval_steps = float(row.get(f"{method}_branch_eval_steps_total", selected_steps))
        eval_count = float(row.get(f"{method}_branch_eval_count", 1.0))
    return {
        "success": success,
        "selected_steps": selected_steps,
        "branch_eval_steps": max(eval_steps, 1.0),
        "branch_eval_count": eval_count,
        "selected_success_per_10_steps": 10.0 * success / max(selected_steps, 1.0),
        "branch_cost_success_per_100_steps": 100.0 * success / max(eval_steps, 1.0),
    }


def add_ranks(records: list[dict[str, Any]]) -> None:
    for value_name, rank_name in (
        ("success_rate", "success_rank"),
        ("selected_success_per_10_steps", "selected_efficiency_rank"),
        ("branch_cost_success_per_100_steps", "branch_cost_efficiency_rank"),
    ):
        ranked = sorted(records, key=lambda item: (-float(item[value_name]), item["method_label"]))
        previous_value: float | None = None
        previous_rank = 0
        for index, record in enumerate(ranked, start=1):
            value = float(record[value_name])
            if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                previous_value = value
                previous_rank = index
            record[rank_name] = previous_rank


def render_svg(path: Path, records: list[dict[str, Any]], value_name: str, title: str) -> None:
    width = 980
    row_height = 34
    top = 72
    left = 260
    right = 42
    height = top + row_height * len(records) + 48
    max_value = max(float(record[value_name]) for record in records) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<text x="24" y="58" font-family="Arial, sans-serif" font-size="12" fill="#475569">Higher is better. Branch-cost metric charges every evaluated branch.</text>',
    ]
    for index, record in enumerate(sorted(records, key=lambda item: -float(item[value_name]))):
        y = top + index * row_height
        value = float(record[value_name])
        bar_width = (width - left - right) * value / max_value
        color = "#0f766e" if record["role"] == "ours" else "#64748b" if record["role"] == "baseline" else "#7c3aed"
        lines.extend(
            [
                f'<text x="24" y="{y + 21}" font-family="Arial, sans-serif" font-size="13">{html.escape(record["method_label"])}</text>',
                f'<rect x="{left}" y="{y + 6}" width="{bar_width:.2f}" height="18" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_width + 8:.2f}" y="{y + 21}" font-family="Arial, sans-serif" font-size="12" fill="#0f172a">{value:.3f}</text>',
            ]
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--selected-svg", type=Path, required=True)
    parser.add_argument("--branch-cost-svg", type=Path, required=True)
    args = parser.parse_args()

    report = load_json(args.summary)
    rows = report["rows"]
    per_method_values: dict[str, list[dict[str, float]]] = {}
    records: list[dict[str, Any]] = []
    for method_key, method_label, role in METHOD_SPECS:
        values = [method_values(row, method_key) for row in rows]
        per_method_values[method_key] = values
        records.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "role": role,
                "unit_count": len(values),
                "success_rate": mean(item["success"] for item in values),
                "mean_selected_steps": mean(item["selected_steps"] for item in values),
                "mean_branch_eval_steps": mean(item["branch_eval_steps"] for item in values),
                "mean_branch_eval_count": mean(item["branch_eval_count"] for item in values),
                "selected_success_per_10_steps": mean(item["selected_success_per_10_steps"] for item in values),
                "branch_cost_success_per_100_steps": mean(item["branch_cost_success_per_100_steps"] for item in values),
            }
        )
    add_ranks(records)

    ours_key = "source_horizon_aware_full"
    comparisons = []
    for method_key, _method_label, role in METHOD_SPECS:
        if method_key == ours_key:
            continue
        comparisons.append(
            {
                "baseline_key": method_key,
                "baseline_role": role,
                "success_mean_delta": mean(
                    ours["success"] - other["success"]
                    for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                ),
                "success_sign_test": sign_test(
                    [
                        ours["success"] - other["success"]
                        for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                    ]
                ),
                "selected_efficiency_mean_delta": mean(
                    ours["selected_success_per_10_steps"] - other["selected_success_per_10_steps"]
                    for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                ),
                "selected_efficiency_sign_test": sign_test(
                    [
                        ours["selected_success_per_10_steps"] - other["selected_success_per_10_steps"]
                        for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                    ]
                ),
                "branch_cost_efficiency_mean_delta": mean(
                    ours["branch_cost_success_per_100_steps"] - other["branch_cost_success_per_100_steps"]
                    for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                ),
                "branch_cost_efficiency_sign_test": sign_test(
                    [
                        ours["branch_cost_success_per_100_steps"] - other["branch_cost_success_per_100_steps"]
                        for ours, other in zip(per_method_values[ours_key], per_method_values[method_key], strict=True)
                    ]
                ),
            }
        )

    output = {
        "experiment_id": "L7-ALFWORLD-COMPUTE-COST-AUDIT",
        "summary": str(args.summary),
        "status": {
            "ours_success_rank_1": next(record for record in records if record["method_key"] == ours_key)["success_rank"] == 1,
            "ours_selected_efficiency_rank_1": next(record for record in records if record["method_key"] == ours_key)["selected_efficiency_rank"] == 1,
            "ours_branch_cost_efficiency_rank_1": next(record for record in records if record["method_key"] == ours_key)["branch_cost_efficiency_rank"] == 1,
            "ours_vs_baseline_branch_race_success_significant_0_05": next(
                comparison
                for comparison in comparisons
                if comparison["baseline_key"] == "baseline_branch_race_with_defer"
            )["success_sign_test"]["significant_0_05"],
            "cost_metric_caveat": (
                "Branch-cost efficiency charges every evaluated branch. It is an audit metric, not the main paper metric, "
                "because branch-race can be implemented with early stopping or parallel execution."
            ),
        },
        "records": records,
        "comparisons": comparisons,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    render_svg(args.selected_svg, records, "selected_success_per_10_steps", "ALFWorld Selected-Branch Efficiency")
    render_svg(args.branch_cost_svg, records, "branch_cost_success_per_100_steps", "ALFWorld Branch-Cost Efficiency Audit")
    print(json.dumps(output["status"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
