#!/usr/bin/env python3
"""Build a local paper-style PK dashboard across current TRACE-H evidence blocks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - scipy is expected in the local env.
    binomtest = None


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
DESIGN_METHODS = [
    ("natural_replan", "natural REPLAN", "baseline"),
    ("anti_loop_retry", "anti-loop retry", "baseline"),
    ("precondition_check", "precondition check", "baseline"),
    ("subgoal_ledger", "subgoal ledger", "baseline"),
    ("bundle_conservative", "TRACE-H bundle conservative", "ours"),
]
TRANSPORT_METHODS = [
    ("best_fixed_source_mean", "best fixed source mean", "baseline"),
    ("fixed_bundle", "fixed bundle", "baseline"),
    ("knn_response", "kNN response transfer", "baseline"),
    ("balanced_ot_lcb", "balanced OT-LCB", "baseline"),
    ("partial_ot_lcb", "TRACE-H partial OT-LCB", "ours"),
]
WEBSHOP_METHODS = [
    ("title_overlap", "title lexical overlap", "baseline"),
    ("description_overlap", "description lexical overlap", "baseline"),
    ("category_query_overlap", "category/query lexical overlap", "baseline"),
    ("all_text_overlap", "all-text lexical overlap", "baseline"),
    ("attribute_overlap", "attribute constraint overlap", "baseline"),
    ("rarest_attribute_anchor", "rarest-attribute anchor", "baseline"),
    ("bm25_all_text", "BM25 all text", "baseline"),
    ("bm25_title_category", "BM25 title+category", "baseline"),
    ("traceh_constraint_query_ledger", "TRACE-H constraint+query ledger", "ours"),
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


def sign_test(diffs: list[float], eps: float = 1e-12) -> dict[str, Any]:
    wins = sum(diff > eps for diff in diffs)
    losses = sum(diff < -eps for diff in diffs)
    ties = len(diffs) - wins - losses
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    elif binomtest is not None:
        p_value = float(binomtest(min(wins, losses), trials, 0.5, alternative="two-sided").pvalue)
    else:
        p_value = float(2.0 * sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / (2**trials))
        p_value = min(p_value, 1.0)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "trials": trials,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def bootstrap_ci(values: list[float], *, rng: np.random.Generator, samples: int = 4000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return 0.0, 0.0
    if np.allclose(array, array[0]):
        value = float(array[0])
        return value, value
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def make_metric_record(
    *,
    dataset: str,
    method_key: str,
    method_label: str,
    role: str,
    primary_metric: str,
    primary_values: list[float],
    secondary_metric: str,
    secondary_values: list[float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    primary_ci = bootstrap_ci(primary_values, rng=rng)
    secondary_ci = bootstrap_ci(secondary_values, rng=rng)
    return {
        "dataset": dataset,
        "method_key": method_key,
        "method_label": method_label,
        "role": role,
        "unit_count": len(primary_values),
        "primary_metric": primary_metric,
        "primary_value": float(mean(primary_values)),
        "primary_ci95": list(primary_ci),
        "secondary_metric": secondary_metric,
        "secondary_value": float(mean(secondary_values)),
        "secondary_ci95": list(secondary_ci),
    }


def alfworld_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = report["rows"]
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in ALFWORLD_METHODS:
        if method_key == "none":
            primary_values = [float(row["none_success"]) for row in rows]
            step_values = [float(row["none_steps"]) for row in rows]
        else:
            primary_values = [float(row[f"{method_key}_success"]) for row in rows]
            step_values = [float(row[f"{method_key}_macro_steps"]) for row in rows]
        secondary_values = [
            10.0 * success / max(steps, 1.0)
            for success, steps in zip(primary_values, step_values, strict=True)
        ]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="ALFWorld dense prefixes",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="success_rate",
                primary_values=primary_values,
                secondary_metric="success_per_10_steps",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def design_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in DESIGN_METHODS:
        primary_values = [
            float(seed_row["summary"]["variants"][method_key]["mean_utility"])
            for seed_row in report["seed_summaries"]
        ]
        secondary_values = [
            float(seed_row["summary"]["variants"][method_key]["accuracy"])
            for seed_row in report["seed_summaries"]
        ]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="Synthetic mechanism design",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="mean_utility",
                primary_values=primary_values,
                secondary_metric="accuracy",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def transport_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    scenario_names = list(report["seed_summaries"])
    seeds = [row["seed"] for row in report["seed_summaries"][scenario_names[0]]]
    by_scenario_seed = {
        scenario: {row["seed"]: row["methods"] for row in rows}
        for scenario, rows in report["seed_summaries"].items()
    }
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in TRANSPORT_METHODS:
        primary_values = []
        secondary_values = []
        for seed in seeds:
            utilities = [
                float(by_scenario_seed[scenario][seed][method_key]["mean_utility"])
                for scenario in scenario_names
            ]
            safety_rates = [
                1.0 - float(by_scenario_seed[scenario][seed][method_key]["negative_rate"])
                for scenario in scenario_names
            ]
            primary_values.append(float(mean(utilities)))
            secondary_values.append(float(mean(safety_rates)))
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="Synthetic transport stress",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="mean_utility",
                primary_values=primary_values,
                secondary_metric="safety_rate",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def webshop_static_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = report["rows"]
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in WEBSHOP_METHODS:
        primary_values = [float(row["methods"][method_key]["exact_top1"]) for row in rows]
        secondary_values = [float(row["methods"][method_key]["reciprocal_rank"]) for row in rows]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="WebShop-small static product selection",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="exact_target_rate",
                primary_values=primary_values,
                secondary_metric="target_mrr",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def webshop_interactive_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = report["rows"]
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in WEBSHOP_INTERACTIVE_METHODS:
        primary_values = [float(row["methods"][method_key]["reward"]) for row in rows]
        secondary_values = [float(row["methods"][method_key]["exact_purchase"]) for row in rows]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="WebShop-small interactive text",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="mean_webshop_reward",
                primary_values=primary_values,
                secondary_metric="exact_purchase_rate",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def harnessbench_routing_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = report["rows"]
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in HARNESSBENCH_ROUTING_METHODS:
        primary_values = [float(row["methods"][method_key]["jaccard"]) for row in rows]
        secondary_values = [float(row["methods"][method_key]["exact_set"]) for row in rows]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="HarnessBench routing projection",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="capability_jaccard",
                primary_values=primary_values,
                secondary_metric="exact_capability_set_rate",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def harnessbench_endpoint_records(report: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = report["records"]
    records = []
    primary_by_method: dict[str, list[float]] = {}
    for method_key, method_label, role in HARNESSBENCH_ENDPOINT_METHODS:
        method_rows = [row for row in rows if row["method_key"] == method_key]
        primary_values = [float(row["outcome_score"]) for row in method_rows]
        secondary_values = [float(row["check_pass_rate"]) for row in method_rows]
        primary_by_method[method_key] = primary_values
        records.append(
            make_metric_record(
                dataset="HarnessBench endpoint outcome subset",
                method_key=method_key,
                method_label=method_label,
                role=role,
                primary_metric="oracle_outcome_score",
                primary_values=primary_values,
                secondary_metric="oracle_check_pass_rate",
                secondary_values=secondary_values,
                rng=rng,
            )
        )
    return records, primary_by_method


def add_ranks(records: list[dict[str, Any]]) -> None:
    for dataset in sorted({record["dataset"] for record in records}):
        dataset_records = [record for record in records if record["dataset"] == dataset]
        for value_name, rank_name in (
            ("primary_value", "primary_rank"),
            ("secondary_value", "secondary_rank"),
        ):
            ranked = sorted(dataset_records, key=lambda item: (-item[value_name], item["method_label"]))
            previous_value: float | None = None
            previous_rank = 0
            for index, record in enumerate(ranked, start=1):
                value = float(record[value_name])
                if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                    previous_rank = index
                    previous_value = value
                record[rank_name] = previous_rank


def comparison_records(
    dataset: str,
    primary_by_method: dict[str, list[float]],
    ours_key: str,
    method_specs: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    ours = primary_by_method[ours_key]
    comparisons = []
    for method_key, method_label, role in method_specs:
        if method_key == ours_key:
            continue
        baseline = primary_by_method[method_key]
        diffs = [a - b for a, b in zip(ours, baseline, strict=True)]
        test = sign_test(diffs)
        comparisons.append(
            {
                "dataset": dataset,
                "ours_key": ours_key,
                "baseline_key": method_key,
                "baseline_label": method_label,
                "baseline_role": role,
                "mean_delta": float(mean(diffs)),
                **test,
            }
        )
    return comparisons


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
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
            writer.writerow({name: record[name] for name in fieldnames})


def render_svg(path: Path, records: list[dict[str, Any]], metric_name: str, value_name: str) -> None:
    datasets = sorted({record["dataset"] for record in records})
    width = 1320
    margin_left = 250
    margin_right = 80
    bar_height = 20
    row_gap = 9
    panel_heights = {
        dataset: 58 + (bar_height + row_gap) * len([record for record in records if record["dataset"] == dataset])
        for dataset in datasets
    }
    height = 70 + sum(panel_heights.values())
    colors = {
        "ours": "#1f77b4",
        "baseline": "#7f7f7f",
        "ablation": "#b7791f",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(metric_name)}</text>',
    ]
    y = 72
    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset"] == dataset]
        max_value = max(record[value_name] for record in dataset_records)
        scale_max = max(1.0, max_value)
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="17" font-weight="700">{html.escape(dataset)}</text>')
        y += 28
        for record in sorted(dataset_records, key=lambda item: item["primary_rank"]):
            value = float(record[value_name])
            bar_width = (width - margin_left - margin_right) * value / scale_max
            color = colors.get(record["role"], "#999999")
            label = f'{record["method_label"]} ({record["role"]})'
            lines.append(
                f'<text x="34" y="{y + 15}" font-family="Arial" font-size="13">{html.escape(label)}</text>'
            )
            lines.append(
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="{color}" rx="2"/>'
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
    parser.add_argument("--design-summary", type=Path, required=True)
    parser.add_argument("--transport-summary", type=Path, required=True)
    parser.add_argument("--webshop-static-summary", type=Path, default=None)
    parser.add_argument("--webshop-interactive-summary", type=Path, default=None)
    parser.add_argument("--harnessbench-routing-summary", type=Path, default=None)
    parser.add_argument("--harnessbench-endpoint-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--primary-svg", type=Path, required=True)
    parser.add_argument("--secondary-svg", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(20260714)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    alf_records, alf_primary = alfworld_records(load_json(args.alfworld_summary), rng)
    design_metric_records, design_primary = design_records(load_json(args.design_summary), rng)
    transport_metric_records, transport_primary = transport_records(load_json(args.transport_summary), rng)
    records.extend(alf_records)
    records.extend(design_metric_records)
    records.extend(transport_metric_records)
    webshop_report = None
    if args.webshop_static_summary is not None:
        webshop_report = load_json(args.webshop_static_summary)
        webshop_metric_records, webshop_primary = webshop_static_records(webshop_report, rng)
        records.extend(webshop_metric_records)
    webshop_interactive_report = None
    if args.webshop_interactive_summary is not None:
        webshop_interactive_report = load_json(args.webshop_interactive_summary)
        webshop_interactive_metric_records, webshop_interactive_primary = webshop_interactive_records(
            webshop_interactive_report,
            rng,
        )
        records.extend(webshop_interactive_metric_records)
    harnessbench_report = None
    harnessbench_endpoint_report = None
    if args.harnessbench_routing_summary is not None:
        harnessbench_report = load_json(args.harnessbench_routing_summary)
        harnessbench_metric_records, harnessbench_primary = harnessbench_routing_records(harnessbench_report, rng)
        records.extend(harnessbench_metric_records)
    if args.harnessbench_endpoint_summary is not None:
        harnessbench_endpoint_report = load_json(args.harnessbench_endpoint_summary)
        harnessbench_endpoint_metric_records, harnessbench_endpoint_primary = harnessbench_endpoint_records(
            harnessbench_endpoint_report,
            rng,
        )
        records.extend(harnessbench_endpoint_metric_records)
    add_ranks(records)
    comparisons.extend(
        comparison_records("ALFWorld dense prefixes", alf_primary, "source_horizon_aware_full", ALFWORLD_METHODS)
    )
    comparisons.extend(
        comparison_records("Synthetic mechanism design", design_primary, "bundle_conservative", DESIGN_METHODS)
    )
    comparisons.extend(
        comparison_records("Synthetic transport stress", transport_primary, "partial_ot_lcb", TRANSPORT_METHODS)
    )
    if args.webshop_static_summary is not None:
        comparisons.extend(
            comparison_records(
                "WebShop-small static product selection",
                webshop_primary,
                "traceh_constraint_query_ledger",
                WEBSHOP_METHODS,
            )
        )
    if args.webshop_interactive_summary is not None:
        comparisons.extend(
            comparison_records(
                "WebShop-small interactive text",
                webshop_interactive_primary,
                "traceh_constraint_query_ledger",
                WEBSHOP_INTERACTIVE_METHODS,
            )
        )
    if args.harnessbench_routing_summary is not None:
        comparisons.extend(
            comparison_records(
                "HarnessBench routing projection",
                harnessbench_primary,
                "traceh_multisource_ledger",
                HARNESSBENCH_ROUTING_METHODS,
            )
        )
    if args.harnessbench_endpoint_summary is not None:
        comparisons.extend(
            comparison_records(
                "HarnessBench endpoint outcome subset",
                harnessbench_endpoint_primary,
                "traceh_endpoint_ledger",
                HARNESSBENCH_ENDPOINT_METHODS,
            )
        )

    datasets = sorted({record["dataset"] for record in records})
    baseline_keys = sorted(
        {
            (record["dataset"], record["method_key"])
            for record in records
            if record["role"] == "baseline"
        }
    )
    ablation_keys = sorted(
        {
            (record["dataset"], record["method_key"])
            for record in records
            if record["role"] == "ablation"
        }
    )
    paper_baseline_comparisons = [
        comparison for comparison in comparisons if comparison["baseline_role"] == "baseline"
    ]
    all_ours_ranked_first = all(
        record["primary_rank"] == 1
        for record in records
        if record["role"] == "ours"
    )
    all_ours_unique_ranked_first = True
    for record in records:
        if record["role"] != "ours":
            continue
        tied_first = [
            other
            for other in records
            if other["dataset"] == record["dataset"]
            and other["primary_rank"] == 1
            and other["method_key"] != record["method_key"]
        ]
        if tied_first:
            all_ours_unique_ranked_first = False
    all_paper_baseline_p_lt_05 = all(
        comparison["significant_0_05"] and comparison["mean_delta"] > 0
        for comparison in paper_baseline_comparisons
    )
    alfworld_independent_baseline_count = sum(
        1 for method_key, _label, role in ALFWORLD_METHODS if role == "baseline" and method_key != "none"
    )
    baseline_count_requirement_met = alfworld_independent_baseline_count >= 6
    has_webshop = args.webshop_static_summary is not None or args.webshop_interactive_summary is not None
    has_harnessbench_routing = args.harnessbench_routing_summary is not None
    has_harnessbench_endpoint = args.harnessbench_endpoint_summary is not None
    has_harnessbench = has_harnessbench_routing or has_harnessbench_endpoint
    real_data_source_count = 1 + (1 if has_webshop else 0) + (1 if has_harnessbench else 0)
    if not has_webshop:
        why_not_complete = [
            "Only ALFWorld is a real local benchmark in this dashboard; the other two evidence blocks are synthetic stress datasets.",
            "Sealed target task-level results and visualizations across three real datasets are still missing.",
        ]
    elif args.webshop_interactive_summary is not None:
        why_not_complete = [
            (
                "The dashboard now includes ALFWorld dense-prefix replay, WebShop-small interactive text, "
                "and HarnessBench endpoint outcome subset."
                if has_harnessbench_endpoint
                else "The dashboard now includes ALFWorld dense-prefix replay, WebShop-small interactive text, "
                "and HarnessBench routing projection."
                if has_harnessbench_routing
                else "The dashboard now includes two real data sources: ALFWorld dense-prefix replay and WebShop-small interactive text."
            ),
            (
                "HarnessBench endpoint evidence is an offline deterministic subset with hand-written local handlers, not yet a sealed LLM-agent final."
                if has_harnessbench_endpoint
                else
                "HarnessBench is currently a routing projection over real tasks, not end-to-end agent outcome."
                if has_harnessbench_routing
                else "A third real benchmark is still missing; the remaining non-ALFWorld/non-WebShop evidence blocks are synthetic or projection-style stress datasets."
            ),
            "Sealed target task-level results and visualizations across three real interactive datasets are still missing.",
        ]
    else:
        why_not_complete = [
            "The dashboard now includes two real data sources: ALFWorld dense-prefix replay and WebShop-small static product selection.",
            "WebShop-small is a static product-selection projection over a real catalog, not yet a full interactive WebShop environment result.",
            "Sealed target task-level results and visualizations across three real interactive datasets are still missing.",
        ]
    if not baseline_count_requirement_met:
        why_not_complete.append("ALFWorld still has fewer than 6 independent macro baselines.")
    if not all_ours_unique_ranked_first:
        why_not_complete.append("Ours is not yet uniquely rank-1 on every local evidence block because some ablations tie it.")
    if not all_paper_baseline_p_lt_05:
        why_not_complete.append("At least one Ours-vs-baseline comparison is not yet significantly positive at p<0.05.")
    status = {
        "local_dataset_count": len(datasets),
        "real_data_source_count": real_data_source_count,
        "unique_baseline_count": len({method for _dataset, method in baseline_keys}),
        "unique_ablation_count": len({method for _dataset, method in ablation_keys}),
        "alfworld_independent_macro_baseline_count_excluding_none": alfworld_independent_baseline_count,
        "alfworld_baseline_count_requirement_met": baseline_count_requirement_met,
        "all_ours_primary_rank_1": all_ours_ranked_first,
        "all_ours_unique_primary_rank_1": all_ours_unique_ranked_first,
        "all_ours_vs_paper_baselines_significant_0_05": all_paper_baseline_p_lt_05,
        "webshop_static_secondary_significant_0_05": (
            None
            if webshop_report is None
            else bool(webshop_report["status"]["ours_vs_all_baselines_secondary_significant_0_05"])
        ),
        "webshop_interactive_secondary_significant_0_05": (
            None
            if webshop_interactive_report is None
            else bool(webshop_interactive_report["status"]["ours_vs_all_baselines_secondary_significant_0_05"])
        ),
        "harnessbench_routing_secondary_significant_0_05": (
            None
            if harnessbench_report is None
            else bool(harnessbench_report["status"]["ours_vs_all_baselines_secondary_significant_0_05"])
        ),
        "harnessbench_endpoint_secondary_significant_0_05": (
            None
            if harnessbench_endpoint_report is None
            else bool(harnessbench_endpoint_report["status"]["ours_vs_all_baselines_secondary_significant_0_05"])
        ),
        "harnessbench_endpoint_success_rate": (
            None
            if harnessbench_endpoint_report is None
            else float(harnessbench_endpoint_report["status"]["ours_endpoint_success_rate"])
        ),
        "paper_goal_satisfied": False,
        "why_not_complete": why_not_complete,
    }
    output = {
        "experiment_id": "L5-LOCAL-PAPER-PK-DASHBOARD",
        "inputs": {
            "alfworld_summary": str(args.alfworld_summary),
            "design_summary": str(args.design_summary),
            "transport_summary": str(args.transport_summary),
            "webshop_static_summary": None
            if args.webshop_static_summary is None
            else str(args.webshop_static_summary),
            "webshop_interactive_summary": None
            if args.webshop_interactive_summary is None
            else str(args.webshop_interactive_summary),
            "harnessbench_routing_summary": None
            if args.harnessbench_routing_summary is None
            else str(args.harnessbench_routing_summary),
            "harnessbench_endpoint_summary": None
            if args.harnessbench_endpoint_summary is None
            else str(args.harnessbench_endpoint_summary),
        },
        "status": status,
        "records": records,
        "comparisons": comparisons,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.table, records)
    render_svg(args.primary_svg, records, "Primary Metric: Ours vs Baselines", "primary_value")
    render_svg(args.secondary_svg, records, "Secondary Metric: Ours vs Baselines", "secondary_value")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
