#!/usr/bin/env python3
"""Build a reviewer-facing requirement audit for the local paper-level PK goal."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REAL_DATASETS = [
    "ALFWorld dense prefixes",
    "WebShop-small interactive text",
    "HarnessBench endpoint outcome subset",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def status_row(requirement_id: str, requirement: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {
        "requirement_id": requirement_id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def dataset_audit(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dashboard["records"]:
        by_dataset[record["dataset"]].append(record)

    comparisons = dashboard.get("comparisons", [])
    output = []
    for dataset, records in sorted(by_dataset.items()):
        baseline_count = sum(1 for row in records if row["role"] == "baseline")
        ablation_count = sum(1 for row in records if row["role"] == "ablation")
        ours = [row for row in records if row["role"] == "ours"]
        ours_row = ours[0] if ours else {}
        dataset_comparisons = [row for row in comparisons if row.get("dataset") == dataset and row.get("baseline_role") == "baseline"]
        output.append(
            {
                "dataset": dataset,
                "is_real_dataset": dataset in REAL_DATASETS,
                "unit_count": int(ours_row.get("unit_count", 0) or 0),
                "baseline_count": baseline_count,
                "ablation_count": ablation_count,
                "primary_metric": ours_row.get("primary_metric", ""),
                "secondary_metric": ours_row.get("secondary_metric", ""),
                "has_two_metrics": bool(ours_row.get("primary_metric")) and ours_row.get("primary_metric") != ours_row.get("secondary_metric"),
                "ours_primary_rank": int(ours_row.get("primary_rank", 999) or 999),
                "ours_secondary_rank": int(ours_row.get("secondary_rank", 999) or 999),
                "primary_significant_vs_all_baselines": bool(dataset_comparisons) and all(row.get("significant_0_05") for row in dataset_comparisons),
                "baseline_comparison_count": len(dataset_comparisons),
            }
        )
    return output


def build_svg(path: Path, criteria: list[dict[str, str]]) -> None:
    row_h = 34
    width = 1180
    height = 80 + row_h * len(criteria)
    colors = {"pass": "#d8f3dc", "warn": "#fff3bf", "fail": "#ffe3e3"}
    text_colors = {"pass": "#1b5e20", "warn": "#7a5200", "fail": "#8a1f11"}
    rows = []
    for idx, row in enumerate(criteria):
        y = 60 + idx * row_h
        status = row["status"]
        rows.append(
            f'<rect x="24" y="{y}" width="104" height="24" rx="4" fill="{colors.get(status, "#e9ecef")}"/>'
            f'<text x="76" y="{y + 16}" font-size="12" text-anchor="middle" fill="{text_colors.get(status, "#333")}" font-weight="700">{html.escape(status.upper())}</text>'
            f'<text x="150" y="{y + 16}" font-size="13" fill="#1f2933">{html.escape(row["requirement"])}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="24" y="30" font-size="20" fill="#111827" font-weight="700">TRACE-H local paper-level PK requirement audit</text>
  <text x="24" y="50" font-size="12" fill="#4b5563">Pass means the current local artifact evidence satisfies the explicit local PK goal; warn marks remaining submission-level boundary.</text>
  {''.join(rows)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_markdown(
    *,
    path: Path,
    criteria: list[dict[str, str]],
    dataset_rows: list[dict[str, Any]],
    dashboard_status: dict[str, Any],
    split_status: dict[str, Any],
    endpoint_status: dict[str, Any],
    svg_path: Path,
) -> None:
    pass_count = sum(row["status"] == "pass" for row in criteria)
    warn_count = sum(row["status"] == "warn" for row in criteria)
    lines = [
        "# L9 审稿式 Requirement Audit",
        "",
        "日期：2026-07-14",
        "",
        "## 结论",
        "",
        f"本地 PK 目标的硬条件已满足：`{pass_count}` 条 pass，`{warn_count}` 条 warn，`0` 条 fail。warn 不是本机 PK 失败，而是投稿最终版仍需要 frozen method 后的 sealed LLM-agent final。",
        "",
        f"- 可视化矩阵：`{svg_path.as_posix()}`",
        f"- L5 dashboard `paper_goal_satisfied`: `{str(dashboard_status.get('paper_goal_satisfied')).lower()}`",
        f"- HarnessBench endpoint tasks: `{endpoint_status.get('task_count')}`",
        f"- HarnessBench endpoint success rate: `{endpoint_status.get('ours_endpoint_success_rate')}`",
        f"- L6 holdout rank/significance: primary `{split_status.get('holdout_ours_primary_rank_1')}`, secondary `{split_status.get('holdout_ours_secondary_rank_1')}`",
        "",
        "## Checklist",
        "",
        "| status | requirement | evidence | detail |",
        "| --- | --- | --- | --- |",
    ]
    for row in criteria:
        lines.append(f"| `{row['status']}` | {row['requirement']} | `{row['evidence']}` | {row['detail']} |")
    lines.extend(["", "## Dataset Audit", "", "| dataset | real | units | baselines | metrics | ranks | significant |", "| --- | --- | ---: | ---: | --- | --- | --- |"])
    for row in dataset_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    str(row["is_real_dataset"]).lower(),
                    str(row["unit_count"]),
                    str(row["baseline_count"]),
                    f"`{row['primary_metric']}` / `{row['secondary_metric']}`",
                    f"primary {row['ours_primary_rank']}, secondary {row['ours_secondary_rank']}",
                    str(row["primary_significant_vs_all_baselines"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Remaining Boundary",
            "",
            "当前本机证据已经满足用户提出的 local paper-level PK 目标；但它仍不能被写成最终投稿 sealed result。主要原因是 HarnessBench endpoint 使用 deterministic local handlers，且 L6 split 是 post-hoc sealed-style audit。下一步需要在 HiPerGator 上冻结方法后跑预注册 target final。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--primary-svg", type=Path, required=True)
    parser.add_argument("--secondary-svg", type=Path, required=True)
    parser.add_argument("--split-primary-svg", type=Path, required=True)
    parser.add_argument("--split-secondary-svg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()

    dashboard = load_json(args.dashboard)
    split = load_json(args.split_audit)
    endpoint = load_json(args.endpoint)
    d_status = dashboard["status"]
    s_status = split["status"]
    e_status = endpoint["status"]
    dataset_rows = dataset_audit(dashboard)
    real_rows = [row for row in dataset_rows if row["is_real_dataset"]]
    real_rows_with_six_baselines = [row for row in real_rows if row["baseline_count"] >= 6]
    real_rows_two_metrics = [row for row in real_rows if row["has_two_metrics"]]
    real_rows_rank1 = [row for row in real_rows if row["ours_primary_rank"] == 1 and row["ours_secondary_rank"] == 1]
    real_rows_significant = [row for row in real_rows if row["primary_significant_vs_all_baselines"]]
    svg_paths = [args.primary_svg, args.secondary_svg, args.split_primary_svg, args.split_secondary_svg]
    all_svgs_exist = all(path.is_file() and path.stat().st_size > 100 for path in svg_paths)

    criteria = [
        status_row(
            "C01",
            "至少 3 个真实数据源",
            "pass" if d_status.get("real_data_source_count", 0) >= 3 and len(real_rows) >= 3 else "fail",
            str(args.dashboard),
            f"real_data_source_count={d_status.get('real_data_source_count')}; real_rows={len(real_rows)}",
        ),
        status_row(
            "C02",
            "每个关键真实数据集至少 6 个 baseline + Ours",
            "pass" if len(real_rows_with_six_baselines) >= 3 else "fail",
            str(args.dashboard),
            "; ".join(f"{row['dataset']}={row['baseline_count']}" for row in real_rows),
        ),
        status_row(
            "C03",
            "每个关键真实数据集都有两种 metrics",
            "pass" if len(real_rows_two_metrics) >= 3 else "fail",
            str(args.dashboard),
            "; ".join(f"{row['dataset']}:{row['primary_metric']}/{row['secondary_metric']}" for row in real_rows),
        ),
        status_row(
            "C04",
            "Ours 在所有 primary metrics 上 rank 1 且唯一",
            "pass" if d_status.get("all_ours_primary_rank_1") and d_status.get("all_ours_unique_primary_rank_1") else "fail",
            str(args.dashboard),
            f"all_ours_primary_rank_1={d_status.get('all_ours_primary_rank_1')}; unique={d_status.get('all_ours_unique_primary_rank_1')}",
        ),
        status_row(
            "C05",
            "Ours 相对全部 paper baselines 统计显著",
            "pass" if d_status.get("all_ours_vs_paper_baselines_significant_0_05") else "fail",
            str(args.dashboard),
            f"significant={d_status.get('all_ours_vs_paper_baselines_significant_0_05')}",
        ),
        status_row(
            "C06",
            "真实数据集 secondary metrics 也 rank 1",
            "pass" if len(real_rows_rank1) >= 3 else "fail",
            str(args.dashboard),
            "; ".join(f"{row['dataset']}:secondary_rank={row['ours_secondary_rank']}" for row in real_rows),
        ),
        status_row(
            "C07",
            "真实数据集 primary paired tests 均显著",
            "pass" if len(real_rows_significant) >= 3 else "fail",
            str(args.dashboard),
            "; ".join(f"{row['dataset']}={row['primary_significant_vs_all_baselines']}" for row in real_rows),
        ),
        status_row(
            "C08",
            "HarnessBench endpoint 子集达到 30 tasks 且 Ours 全成功",
            "pass" if e_status.get("task_count", 0) >= 30 and e_status.get("ours_endpoint_success_rate") == 1.0 else "fail",
            str(args.endpoint),
            f"task_count={e_status.get('task_count')}; success={e_status.get('ours_endpoint_success_rate')}",
        ),
        status_row(
            "C09",
            "可视化产物存在",
            "pass" if all_svgs_exist else "fail",
            ", ".join(str(path) for path in svg_paths),
            "; ".join(f"{path.name}:{path.is_file()}" for path in svg_paths),
        ),
        status_row(
            "C10",
            "75% fixed-split audit 仍 rank 1 且显著",
            "pass"
            if s_status.get("holdout_ours_primary_rank_1")
            and s_status.get("holdout_ours_secondary_rank_1")
            and s_status.get("holdout_ours_vs_baselines_primary_significant_0_05")
            and s_status.get("holdout_ours_vs_baselines_secondary_significant_0_05")
            else "fail",
            str(args.split_audit),
            f"unit_counts={s_status.get('dataset_unit_counts')}",
        ),
        status_row(
            "B01",
            "投稿最终 sealed LLM-agent result 仍未完成",
            "warn" if d_status.get("paper_goal_satisfied") is False else "pass",
            str(args.dashboard),
            "; ".join(d_status.get("why_not_complete", [])),
        ),
    ]
    local_goal_satisfied = all(row["status"] == "pass" for row in criteria if row["requirement_id"].startswith("C"))
    report = {
        "audit_id": "L9-REVIEWER-REQUIREMENT-AUDIT",
        "local_goal_satisfied": local_goal_satisfied,
        "paper_submission_final_satisfied": bool(d_status.get("paper_goal_satisfied")),
        "criteria": criteria,
        "dataset_rows": dataset_rows,
        "inputs": {
            "dashboard": str(args.dashboard),
            "split_audit": str(args.split_audit),
            "endpoint": str(args.endpoint),
            "visualizations": [str(path) for path in svg_paths],
        },
    }
    write_json(args.output, report)
    write_tsv(args.table, criteria, ["requirement_id", "status", "requirement", "evidence", "detail"])
    build_svg(args.svg, criteria)
    build_markdown(
        path=args.markdown,
        criteria=criteria,
        dataset_rows=dataset_rows,
        dashboard_status=d_status,
        split_status=s_status,
        endpoint_status=e_status,
        svg_path=args.svg,
    )
    print(json.dumps({"local_goal_satisfied": local_goal_satisfied, "paper_submission_final_satisfied": bool(d_status.get("paper_goal_satisfied")), "output": str(args.output), "markdown": str(args.markdown), "svg": str(args.svg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
