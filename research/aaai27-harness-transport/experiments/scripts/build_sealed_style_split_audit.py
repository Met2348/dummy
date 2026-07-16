#!/usr/bin/env python3
"""Build a fixed-split audit over local TRACE-H evidence blocks.

This is a post-hoc sealed-style audit, not a true sealed final: some methods
were iterated before this split file existed.  The point is to make the split
rule explicit and reveal which evidence blocks remain strong on a deterministic
holdout subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ALFWORLD_METHODS = [
    ("none", "NONE / source policy", "baseline"),
    ("target_first_full", "target-first macro", "baseline"),
    ("container_first_full", "container-first macro", "baseline"),
    ("reverse_container_full", "reverse-container macro", "baseline"),
    ("kitchen_first_full", "kitchen-first macro", "baseline"),
    ("furniture_first_full", "furniture-first macro", "baseline"),
    ("alphabetical_full", "alphabetical macro", "baseline"),
    ("baseline_branch_race_with_defer", "baseline-only branch race", "baseline"),
    ("source_first_full", "source-first no horizon switch", "ablation"),
    ("source_first_no_history_ledger", "no history ledger", "ablation"),
    ("source_first_no_deposit_lock", "no deposit lock", "ablation"),
    ("source_first_no_inventory_inference", "no inventory inference", "ablation"),
    ("source_first_no_instance_ledger", "no instance ledger", "ablation"),
    ("source_exhaustive_full", "source-prioritized exhaustive sweep", "ablation"),
    ("source_horizon_aware_full", "TRACE-H branch-race ledger", "ours"),
]

WEBSHOP_INTERACTIVE_METHODS = [
    ("random_top10_full_search", "random top-10 after full search", "baseline"),
    ("bm25_full_instruction", "BM25 full instruction", "baseline"),
    ("bm25_attribute_query", "BM25 attribute query", "baseline"),
    ("bm25_core_query", "BM25 core query", "baseline"),
    ("title_overlap_full_search", "title overlap after full search", "baseline"),
    ("description_overlap_full_search", "description overlap after full search", "baseline"),
    ("all_text_overlap_full_search", "all-text overlap after full search", "baseline"),
    ("attribute_overlap_attr_search", "attribute overlap after attr search", "baseline"),
    ("rarest_attribute_anchor", "rarest-attribute anchor", "baseline"),
    ("traceh_constraint_query_ledger", "TRACE-H constraint+query ledger", "ours"),
]

HARNESSBENCH_ROUTING_METHODS = [
    ("majority_prior", "majority prior", "baseline"),
    ("title_only", "title-only keywords", "baseline"),
    ("prompt_only", "prompt-only keywords", "baseline"),
    ("tags_only", "tags-only keywords", "baseline"),
    ("fixtures_only", "fixture-only keywords", "baseline"),
    ("oracle_only", "oracle-only keywords", "baseline"),
    ("title_prompt", "title+prompt keywords", "baseline"),
    ("all_text_flat", "flat all-text keywords", "baseline"),
    ("traceh_multisource_ledger", "TRACE-H multisource routing ledger", "ours"),
]
HARNESSBENCH_ENDPOINT_METHODS = [
    ("no_output", "no output", "baseline"),
    ("schema_only", "schema-only stubs", "baseline"),
    ("prompt_literal_stub", "prompt-literal stub", "baseline"),
    ("fixture_copy", "fixture copy/no transform", "baseline"),
    ("demo_local", "HarnessBench demo adapter", "baseline"),
    ("narrow_file_tool", "narrow file/email tool", "baseline"),
    ("traceh_endpoint_ledger", "TRACE-H endpoint ledger", "ours"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_name(dataset: str, unit_id: str, holdout_rate: float) -> str:
    digest = hashlib.sha256(f"{dataset}::{unit_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "holdout" if value < holdout_rate else "development"


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


def add_ranks(records: list[dict[str, Any]]) -> None:
    for dataset in sorted({record["dataset"] for record in records}):
        for split in sorted({record["split"] for record in records if record["dataset"] == dataset}):
            subset = [record for record in records if record["dataset"] == dataset and record["split"] == split]
            for value_name, rank_name in (("primary_value", "primary_rank"), ("secondary_value", "secondary_rank")):
                ranked = sorted(subset, key=lambda item: (-float(item[value_name]), item["method_label"]))
                previous_value: float | None = None
                previous_rank = 0
                for index, record in enumerate(ranked, start=1):
                    value = float(record[value_name])
                    if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                        previous_value = value
                        previous_rank = index
                    record[rank_name] = previous_rank


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def append_record(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    method_key: str,
    method_label: str,
    role: str,
    primary_metric: str,
    primary_values: list[float],
    secondary_metric: str,
    secondary_values: list[float],
) -> None:
    records.append(
        {
            "dataset": dataset,
            "split": split,
            "method_key": method_key,
            "method_label": method_label,
            "role": role,
            "unit_count": len(primary_values),
            "primary_metric": primary_metric,
            "primary_value": mean_or_zero(primary_values),
            "secondary_metric": secondary_metric,
            "secondary_value": mean_or_zero(secondary_values),
        }
    )


def alfworld_units(report: dict[str, Any], holdout_rate: float) -> list[dict[str, Any]]:
    units = []
    for index, row in enumerate(report["rows"]):
        unit_id = str(row.get("prefix_id") or row.get("case_id") or row.get("task_id") or index)
        methods = {}
        for method_key, _label, _role in ALFWORLD_METHODS:
            if method_key == "none":
                success = float(row["none_success"])
                steps = float(row["none_steps"])
            else:
                success = float(row[f"{method_key}_success"])
                steps = float(row[f"{method_key}_macro_steps"])
            methods[method_key] = {
                "primary": success,
                "secondary": 10.0 * success / max(1.0, steps),
            }
        units.append(
            {
                "unit_id": unit_id,
                "split": split_name("ALFWorld dense prefixes", unit_id, holdout_rate),
                "methods": methods,
            }
        )
    return units


def webshop_units(report: dict[str, Any], holdout_rate: float) -> list[dict[str, Any]]:
    units = []
    for row in report["rows"]:
        unit_id = str(row["task_index"])
        units.append(
            {
                "unit_id": unit_id,
                "split": split_name("WebShop-small interactive text", unit_id, holdout_rate),
                "methods": {
                    method_key: {
                        "primary": float(row["methods"][method_key]["reward"]),
                        "secondary": float(row["methods"][method_key]["exact_purchase"]),
                    }
                    for method_key, _label, _role in WEBSHOP_INTERACTIVE_METHODS
                },
            }
        )
    return units


def harnessbench_units(report: dict[str, Any], holdout_rate: float) -> list[dict[str, Any]]:
    units = []
    for row in report["rows"]:
        unit_id = str(row["task_id"])
        units.append(
            {
                "unit_id": unit_id,
                "split": split_name("HarnessBench routing projection", unit_id, holdout_rate),
                "methods": {
                    method_key: {
                        "primary": float(row["methods"][method_key]["jaccard"]),
                        "secondary": float(row["methods"][method_key]["exact_set"]),
                    }
                    for method_key, _label, _role in HARNESSBENCH_ROUTING_METHODS
                },
            }
        )
    return units


def harnessbench_endpoint_units(report: dict[str, Any], holdout_rate: float) -> list[dict[str, Any]]:
    units = []
    task_ids = sorted({str(row["task_id"]) for row in report["records"]})
    for task_id in task_ids:
        task_rows = [row for row in report["records"] if str(row["task_id"]) == task_id]
        by_method = {row["method_key"]: row for row in task_rows}
        units.append(
            {
                "unit_id": task_id,
                "split": split_name("HarnessBench endpoint outcome subset", task_id, holdout_rate),
                "methods": {
                    method_key: {
                        "primary": float(by_method[method_key]["outcome_score"]),
                        "secondary": float(by_method[method_key]["check_pass_rate"]),
                    }
                    for method_key, _label, _role in HARNESSBENCH_ENDPOINT_METHODS
                },
            }
        )
    return units


def dataset_records(
    *,
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    dataset: str,
    units: list[dict[str, Any]],
    method_specs: list[tuple[str, str, str]],
    ours_key: str,
    primary_metric: str,
    secondary_metric: str,
) -> None:
    for split in ("development", "holdout", "all"):
        split_units = units if split == "all" else [unit for unit in units if unit["split"] == split]
        for method_key, method_label, role in method_specs:
            primary_values = [float(unit["methods"][method_key]["primary"]) for unit in split_units]
            secondary_values = [float(unit["methods"][method_key]["secondary"]) for unit in split_units]
            append_record(
                records,
                dataset=dataset,
                split=split,
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric=primary_metric,
                primary_values=primary_values,
                secondary_metric=secondary_metric,
                secondary_values=secondary_values,
            )
        for method_key, method_label, role in method_specs:
            if method_key == ours_key:
                continue
            primary_diffs = [
                float(unit["methods"][ours_key]["primary"]) - float(unit["methods"][method_key]["primary"])
                for unit in split_units
            ]
            secondary_diffs = [
                float(unit["methods"][ours_key]["secondary"]) - float(unit["methods"][method_key]["secondary"])
                for unit in split_units
            ]
            comparisons.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "ours_key": ours_key,
                    "baseline_key": method_key,
                    "baseline_label": method_label,
                    "baseline_role": role,
                    "unit_count": len(split_units),
                    "primary_mean_delta": mean_or_zero(primary_diffs),
                    "secondary_mean_delta": mean_or_zero(secondary_diffs),
                    "primary_sign_test": sign_test(primary_diffs),
                    "secondary_sign_test": sign_test(secondary_diffs),
                }
            )


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "method_label",
        "role",
        "unit_count",
        "primary_metric",
        "primary_value",
        "primary_rank",
        "secondary_metric",
        "secondary_value",
        "secondary_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def render_svg(path: Path, records: list[dict[str, Any]], value_name: str, title: str) -> None:
    holdout_records = [record for record in records if record["split"] == "holdout"]
    datasets = sorted({record["dataset"] for record in holdout_records})
    width = 1320
    margin_left = 280
    margin_right = 80
    bar_height = 20
    row_gap = 9
    panel_heights = {
        dataset: 58 + (bar_height + row_gap) * len([record for record in holdout_records if record["dataset"] == dataset])
        for dataset in datasets
    }
    height = 70 + sum(panel_heights.values())
    colors = {"ours": "#1f77b4", "baseline": "#7f7f7f", "ablation": "#b7791f"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    y = 72
    for dataset in datasets:
        dataset_records = [record for record in holdout_records if record["dataset"] == dataset]
        max_value = max(float(record[value_name]) for record in dataset_records) if dataset_records else 1.0
        scale_max = max(1.0, max_value)
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="17" font-weight="700">{html.escape(dataset)}</text>')
        y += 28
        for record in sorted(dataset_records, key=lambda item: item[f"{value_name.split('_')[0]}_rank"]):
            value = float(record[value_name])
            bar_width = (width - margin_left - margin_right) * value / scale_max
            label = f'{record["method_label"]} ({record["role"]})'
            lines.append(f'<text x="34" y="{y + 15}" font-family="Arial" font-size="13">{html.escape(label)}</text>')
            lines.append(
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="{colors.get(record["role"], "#999999")}" rx="2"/>'
            )
            lines.append(
                f'<text x="{margin_left + bar_width + 8:.2f}" y="{y + 15}" font-family="Arial" font-size="13">{value:.3f}</text>'
            )
            y += bar_height + row_gap
        y += 25
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld-summary", type=Path, required=True)
    parser.add_argument("--webshop-interactive-summary", type=Path, required=True)
    parser.add_argument("--harnessbench-routing-summary", type=Path, default=None)
    parser.add_argument("--harnessbench-endpoint-summary", type=Path, default=None)
    parser.add_argument("--holdout-rate", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--primary-svg", type=Path, required=True)
    parser.add_argument("--secondary-svg", type=Path, required=True)
    args = parser.parse_args()
    if args.harnessbench_endpoint_summary is None and args.harnessbench_routing_summary is None:
        raise SystemExit("provide --harnessbench-endpoint-summary or --harnessbench-routing-summary")

    if args.harnessbench_endpoint_summary is not None:
        harnessbench_dataset = "HarnessBench endpoint outcome subset"
        harnessbench_units_list = harnessbench_endpoint_units(load_json(args.harnessbench_endpoint_summary), args.holdout_rate)
        harnessbench_method_specs = HARNESSBENCH_ENDPOINT_METHODS
        harnessbench_ours_key = "traceh_endpoint_ledger"
        harnessbench_primary_metric = "oracle_outcome_score"
        harnessbench_secondary_metric = "oracle_check_pass_rate"
    else:
        harnessbench_dataset = "HarnessBench routing projection"
        harnessbench_units_list = harnessbench_units(load_json(args.harnessbench_routing_summary), args.holdout_rate)
        harnessbench_method_specs = HARNESSBENCH_ROUTING_METHODS
        harnessbench_ours_key = "traceh_multisource_ledger"
        harnessbench_primary_metric = "capability_jaccard"
        harnessbench_secondary_metric = "exact_capability_set_rate"

    units_by_dataset = {
        "ALFWorld dense prefixes": alfworld_units(load_json(args.alfworld_summary), args.holdout_rate),
        "WebShop-small interactive text": webshop_units(load_json(args.webshop_interactive_summary), args.holdout_rate),
        harnessbench_dataset: harnessbench_units_list,
    }
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    dataset_records(
        records=records,
        comparisons=comparisons,
        dataset="ALFWorld dense prefixes",
        units=units_by_dataset["ALFWorld dense prefixes"],
        method_specs=ALFWORLD_METHODS,
        ours_key="source_horizon_aware_full",
        primary_metric="success_rate",
        secondary_metric="success_per_10_steps",
    )
    dataset_records(
        records=records,
        comparisons=comparisons,
        dataset="WebShop-small interactive text",
        units=units_by_dataset["WebShop-small interactive text"],
        method_specs=WEBSHOP_INTERACTIVE_METHODS,
        ours_key="traceh_constraint_query_ledger",
        primary_metric="mean_webshop_reward",
        secondary_metric="exact_purchase_rate",
    )
    dataset_records(
        records=records,
        comparisons=comparisons,
        dataset=harnessbench_dataset,
        units=units_by_dataset[harnessbench_dataset],
        method_specs=harnessbench_method_specs,
        ours_key=harnessbench_ours_key,
        primary_metric=harnessbench_primary_metric,
        secondary_metric=harnessbench_secondary_metric,
    )
    add_ranks(records)

    holdout_baseline_comparisons = [
        comparison
        for comparison in comparisons
        if comparison["split"] == "holdout" and comparison["baseline_role"] == "baseline"
    ]
    holdout_ours_records = [record for record in records if record["split"] == "holdout" and record["role"] == "ours"]
    status = {
        "audit_type": "post_hoc_sealed_style_fixed_hash_split",
        "holdout_rate": args.holdout_rate,
        "dataset_unit_counts": {
            dataset: {
                "all": len(units),
                "holdout": sum(unit["split"] == "holdout" for unit in units),
                "development": sum(unit["split"] == "development" for unit in units),
            }
            for dataset, units in units_by_dataset.items()
        },
        "holdout_ours_primary_rank_1": all(record["primary_rank"] == 1 for record in holdout_ours_records),
        "holdout_ours_secondary_rank_1": all(record["secondary_rank"] == 1 for record in holdout_ours_records),
        "holdout_ours_vs_baselines_primary_significant_0_05": all(
            comparison["primary_sign_test"]["significant_0_05"] and comparison["primary_mean_delta"] > 0
            for comparison in holdout_baseline_comparisons
        ),
        "holdout_ours_vs_baselines_secondary_significant_0_05": all(
            comparison["secondary_sign_test"]["significant_0_05"] and comparison["secondary_mean_delta"] > 0
            for comparison in holdout_baseline_comparisons
        ),
        "not_true_sealed_final_reason": (
            "The split is deterministic and useful for audit, but was introduced after local method iteration; "
            "a true paper sealed final still needs pre-registered target splits."
        ),
    }
    output = {
        "experiment_id": "L6-SEALED-STYLE-SPLIT-AUDIT",
        "inputs": {
            "alfworld_summary": str(args.alfworld_summary),
            "webshop_interactive_summary": str(args.webshop_interactive_summary),
            "harnessbench_routing_summary": None
            if args.harnessbench_routing_summary is None
            else str(args.harnessbench_routing_summary),
            "harnessbench_endpoint_summary": None
            if args.harnessbench_endpoint_summary is None
            else str(args.harnessbench_endpoint_summary),
            "holdout_rate": args.holdout_rate,
        },
        "status": status,
        "records": records,
        "comparisons": comparisons,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.table, records)
    render_svg(args.primary_svg, records, "primary_value", "Holdout Primary Metric: Fixed Split Audit")
    render_svg(args.secondary_svg, records, "secondary_value", "Holdout Secondary Metric: Fixed Split Audit")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
