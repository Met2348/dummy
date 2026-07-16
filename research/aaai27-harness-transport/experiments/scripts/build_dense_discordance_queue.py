#!/usr/bin/env python3
"""Build an audit queue for discordant dense macro PK outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_family(task_id: str) -> str:
    parts = Path(task_id).parts
    for part in parts:
        if part.startswith(("pick_", "look_at_obj", "pick_two_obj")):
            return part
    return Path(task_id).parent.name


def compact_row(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    methods = [
        "target_first_full",
        "source_first_full",
        "source_first_no_history_ledger",
        "source_first_no_deposit_lock",
        "source_first_no_inventory_inference",
        "source_first_no_instance_ledger",
    ]
    output: dict[str, Any] = {
        "block": row["block"],
        "candidate_id": row["candidate_id"],
        "baseline_run_id": row["baseline_run_id"],
        "task_family": task_family(row["task_id"]),
        "task_id": row["task_id"],
        "trace_path": candidate.get("trace_path"),
        "prefix_step": row["prefix_step"],
        "none_success": row["none_success"],
        "none_score": row["none_score"],
        "none_steps": row["none_steps"],
        "revisit_count": candidate.get("revisit_count"),
        "prefix_state_hash": candidate.get("prefix_state_hash"),
    }
    for method in methods:
        output[method] = {
            "success": row[f"{method}_success"],
            "score": row[f"{method}_score"],
            "macro_steps": row[f"{method}_macro_steps"],
            "delivered_count": row[f"{method}_delivered_count"],
            "deposit_target": row[f"{method}_deposit_target"],
        }
    return output


def bucket_rows(rows: list[dict[str, Any]], candidates: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "source_beats_none": [],
        "none_beats_source": [],
        "source_beats_target": [],
        "target_beats_source": [],
        "source_beats_both_none_and_target": [],
        "source_full_lost_by_no_instance_ledger": [],
    }
    for row in rows:
        candidate = candidates[row["candidate_id"]]
        compact = compact_row(row, candidate)
        source = row["source_first_full_success"]
        target = row["target_first_full_success"]
        none = row["none_success"]
        no_instance = row["source_first_no_instance_ledger_success"]
        if source and not none:
            buckets["source_beats_none"].append(compact)
        if none and not source:
            buckets["none_beats_source"].append(compact)
        if source and not target:
            buckets["source_beats_target"].append(compact)
        if target and not source:
            buckets["target_beats_source"].append(compact)
        if source and not none and not target:
            buckets["source_beats_both_none_and_target"].append(compact)
        if source and not no_instance:
            buckets["source_full_lost_by_no_instance_ledger"].append(compact)
    return buckets


def render_markdown(summary: dict[str, Any], buckets: dict[str, list[dict[str, Any]]]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# TRACE-H dense macro PK 分歧审计队列",
        "",
        "- **定位**：development audit queue；不是 sealed target final result。",
        f"- **候选数**：{aggregate['candidate_count']}",
        f"- **source-first vs NONE 独有成功/失败**：{aggregate['source_only_vs_none']} / {aggregate['none_only_vs_source']}",
        f"- **source-first vs target-first 独有成功/失败**：{aggregate['source_only_vs_target']} / {aggregate['target_only_vs_source']}",
        "",
        "## 优先级",
        "",
        "1. 先读 `source_beats_both_none_and_target`：这是最适合写论文 qualitative evidence 的 source-first 独赢样本。",
        "2. 再读 `none_beats_source` 与 `target_beats_source`：这是机制边界与反例，必须解释清楚，不能藏。",
        "3. 最后读 `source_full_lost_by_no_instance_ledger`：这是 instance ledger 必要性的消融证据。",
        "",
    ]
    bucket_titles = {
        "source_beats_both_none_and_target": "source-first 同时击败 NONE 与 target-first",
        "source_beats_none": "source-first 击败 NONE",
        "none_beats_source": "NONE 击败 source-first",
        "source_beats_target": "source-first 击败 target-first",
        "target_beats_source": "target-first 击败 source-first",
        "source_full_lost_by_no_instance_ledger": "去掉 instance ledger 后丢失的 source-first 成功",
    }
    for bucket_name, title in bucket_titles.items():
        items = buckets[bucket_name]
        lines.extend([f"## {title}", "", f"- count: {len(items)}", ""])
        if not items:
            continue
        lines.append("| block | candidate | task family | prefix | NONE | target | source | no-instance | trace |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for item in items:
            trace = item["trace_path"] or ""
            lines.append(
                "| {block} | `{candidate_id}` | `{task_family}` | {prefix_step} | {none} | {target} | {source} | {no_instance} | `{trace}` |".format(
                    block=item["block"],
                    candidate_id=item["candidate_id"],
                    task_family=item["task_family"],
                    prefix_step=item["prefix_step"],
                    none=int(item["none_success"]),
                    target=int(item["target_first_full"]["success"]),
                    source=int(item["source_first_full"]["success"]),
                    no_instance=int(item["source_first_no_instance_ledger"]["success"]),
                    trace=trace,
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    summary = load_json(args.summary)
    candidate_report = load_json(args.candidates)
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in candidate_report["candidates"]
    }
    missing = [row["candidate_id"] for row in summary["rows"] if row["candidate_id"] not in candidates]
    if missing:
        raise ValueError(f"summary rows missing from candidate report: {missing[:5]}")
    buckets = bucket_rows(summary["rows"], candidates)
    output = {
        "experiment_id": "L3-DENSE-MACRO-PK-DISCORDANCE-AUDIT-QUEUE",
        "summary_report": str(args.summary),
        "candidate_report": str(args.candidates),
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "buckets": buckets,
        "notes": [
            "Development audit queue over dense no-progress prefixes.",
            "Use source_beats_both_none_and_target for qualitative mechanism examples.",
            "Audit none_beats_source and target_beats_source before strengthening claims.",
        ],
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary, buckets), encoding="utf-8")
    print(json.dumps(output["bucket_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
