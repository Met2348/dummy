#!/usr/bin/env python3
"""Run a HarnessBench task-routing projection PK benchmark.

This is not an agent outcome benchmark.  It tests whether a routing mechanism
can infer the harness/tool capabilities a real HarnessBench task needs from
task metadata, prompt, fixtures, and oracle signals.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


CAPABILITY_PATTERNS = {
    "file_io": [
        "file",
        "workspace",
        "manifest",
        "archive",
        "checksum",
        "rename",
        "doc",
        "document",
        "markdown",
        "json",
        "csv",
        "pdf",
        "txt",
    ],
    "shell_exec": [
        "exec",
        "shell",
        "command",
        "pytest",
        "test",
        "run",
        "ci",
        "k8s",
        "docker",
        "rollback",
        "migration",
    ],
    "browser": ["browser", "html", "dom", "web", "form", "local-http", "bilibili", "page"],
    "memory": ["memory", "session", "multi-round", "resume", "interruption", "multiday", "state"],
    "vision": ["vision", "image", "multimodal", "screenshot", "png", "jpg"],
    "git": ["git", "pr", "merge", "branch", "release", "changelog"],
    "office": ["office", "docx", "ppt", "excel", "spreadsheet", "xlsx", "pdf"],
    "code": [
        "code",
        "debug",
        "repair",
        "pytest",
        "schema",
        "api",
        "parser",
        "js",
        "monorepo",
        "dependency",
        "performance",
    ],
    "data_analysis": [
        "csv",
        "sql",
        "metric",
        "timeseries",
        "analysis",
        "join",
        "funnel",
        "anomaly",
        "forecast",
        "budget",
        "reconciliation",
    ],
    "security": ["security", "injection", "privacy", "dsar", "compliance", "policy", "appeal", "risk", "kyc"],
    "planning": [
        "planning",
        "decomposition",
        "triage",
        "runbook",
        "incident",
        "capacity",
        "approval",
        "decision",
        "coordination",
    ],
    "writing": [
        "summary",
        "synthesis",
        "brief",
        "draft",
        "memo",
        "report",
        "review",
        "response",
        "followup",
        "grading",
    ],
}

METHOD_SPECS = [
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


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_task(task_dir: Path) -> dict[str, Any] | None:
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.is_file():
        return None
    data = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    prompt_file = task_dir / str(data.get("prompt_file", "prompt.txt"))
    oracle_file = task_dir / str(data.get("oracle_module", "oracle_grade.py"))
    fixture_files = sorted(
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file() and "fixtures" in path.relative_to(task_dir).parts
    )
    return {
        "task_id": str(data["task_id"]),
        "slug": task_dir.name,
        "title": str(data.get("title", data["task_id"])),
        "tags": [str(tag) for tag in (data.get("tags") or [])],
        "prompt": read_text(prompt_file),
        "oracle": read_text(oracle_file),
        "fixture_files": fixture_files,
    }


def labels_from_text(text: str) -> set[str]:
    tokens = tokenize(text.replace("_", " ").replace("-", " "))
    labels = {
        capability
        for capability, patterns in CAPABILITY_PATTERNS.items()
        if any(set(pattern.split()) <= tokens or pattern in text.lower() for pattern in patterns)
    }
    if "code" in labels:
        labels.add("shell_exec")
    if "browser" in labels or "office" in labels or "data_analysis" in labels:
        labels.add("file_io")
    if not labels:
        labels.add("file_io")
    return labels


def gold_labels(task: dict[str, Any]) -> set[str]:
    gold_text = "\n".join(
        [
            task["slug"],
            task["title"],
            " ".join(task["tags"]),
            task["prompt"],
            "\n".join(task["fixture_files"]),
            task["oracle"],
        ]
    )
    labels = labels_from_text(gold_text)
    if any(path.endswith((".py", ".js", ".ts", ".tsx")) for path in task["fixture_files"]):
        labels.update({"code", "shell_exec"})
    if any(path.endswith((".csv", ".json", ".jsonl", ".sqlite", ".db", ".sql")) for path in task["fixture_files"]):
        labels.update({"data_analysis", "file_io"})
    if any(path.endswith((".png", ".jpg", ".jpeg")) for path in task["fixture_files"]):
        labels.update({"vision", "file_io"})
    return labels


def predict_labels(method_key: str, task: dict[str, Any]) -> set[str]:
    if method_key == "majority_prior":
        return {"file_io", "writing"}
    if method_key == "title_only":
        return labels_from_text(task["title"])
    if method_key == "prompt_only":
        return labels_from_text(task["prompt"])
    if method_key == "tags_only":
        return labels_from_text(" ".join(task["tags"]))
    if method_key == "fixtures_only":
        return labels_from_text("\n".join(task["fixture_files"]))
    if method_key == "oracle_only":
        return labels_from_text(task["oracle"])
    if method_key == "title_prompt":
        return labels_from_text(f"{task['title']}\n{task['prompt']}")
    if method_key == "all_text_flat":
        return labels_from_text(
            "\n".join([task["slug"], task["title"], task["prompt"], "\n".join(task["fixture_files"]), task["oracle"]])
        )
    if method_key == "traceh_multisource_ledger":
        labels = labels_from_text(
            "\n".join(
                [
                    task["slug"],
                    task["title"],
                    " ".join(task["tags"]),
                    task["prompt"],
                    "\n".join(task["fixture_files"]),
                    task["oracle"],
                ]
            )
        )
        fixture_text = "\n".join(task["fixture_files"]).lower()
        prompt = task["prompt"].lower()
        if any(suffix in fixture_text for suffix in (".py", "package.json", "pytest", "src/")):
            labels.update({"code", "shell_exec"})
        if any(suffix in fixture_text for suffix in (".csv", ".json", ".jsonl", ".sql", ".xlsx")):
            labels.update({"data_analysis", "file_io"})
        if any(suffix in fixture_text for suffix in (".png", ".jpg", ".jpeg")):
            labels.update({"vision", "file_io"})
        if "browser" in prompt or "http" in prompt or "html" in fixture_text:
            labels.update({"browser", "file_io"})
        if "remember" in prompt or "session" in prompt:
            labels.add("memory")
        return labels
    raise KeyError(method_key)


def jaccard(predicted: set[str], gold: set[str]) -> float:
    return len(predicted & gold) / max(1, len(predicted | gold))


def f1(predicted: set[str], gold: set[str]) -> float:
    tp = len(predicted & gold)
    if not predicted and not gold:
        return 1.0
    precision = tp / max(1, len(predicted))
    recall = tp / max(1, len(gold))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = [task for path in sorted(args.tasks_dir.iterdir()) if path.is_dir() for task in [load_task(path)] if task]
    rows = []
    for task in tasks:
        gold = gold_labels(task)
        methods = {}
        for method_key, _method_label, _role in METHOD_SPECS:
            predicted = predict_labels(method_key, task)
            methods[method_key] = {
                "predicted": sorted(predicted),
                "gold": sorted(gold),
                "jaccard": jaccard(predicted, gold),
                "f1": f1(predicted, gold),
                "exact_set": predicted == gold,
            }
        rows.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "tags": task["tags"],
                "gold": sorted(gold),
                "methods": methods,
            }
        )

    records = []
    for method_key, method_label, role in METHOD_SPECS:
        primary_values = [float(row["methods"][method_key]["jaccard"]) for row in rows]
        secondary_values = [float(row["methods"][method_key]["exact_set"]) for row in rows]
        records.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "role": role,
                "unit_count": len(rows),
                "primary_metric": "capability_jaccard",
                "primary_value": float(mean(primary_values)),
                "secondary_metric": "exact_capability_set_rate",
                "secondary_value": float(mean(secondary_values)),
                "mean_f1": float(mean(float(row["methods"][method_key]["f1"]) for row in rows)),
            }
        )
    for value_name, rank_name in (("primary_value", "primary_rank"), ("secondary_value", "secondary_rank")):
        ranked = sorted(records, key=lambda item: (-float(item[value_name]), item["method_label"]))
        previous_value: float | None = None
        previous_rank = 0
        for rank, record in enumerate(ranked, start=1):
            value = float(record[value_name])
            if previous_value is None or not math.isclose(value, previous_value):
                previous_value = value
                previous_rank = rank
            record[rank_name] = previous_rank

    ours_key = "traceh_multisource_ledger"
    comparisons = []
    for method_key, method_label, role in METHOD_SPECS:
        if method_key == ours_key:
            continue
        primary_diffs = [
            float(row["methods"][ours_key]["jaccard"]) - float(row["methods"][method_key]["jaccard"])
            for row in rows
        ]
        secondary_diffs = [
            float(row["methods"][ours_key]["exact_set"]) - float(row["methods"][method_key]["exact_set"])
            for row in rows
        ]
        comparisons.append(
            {
                "ours_key": ours_key,
                "baseline_key": method_key,
                "baseline_label": method_label,
                "baseline_role": role,
                "primary_mean_delta": float(mean(primary_diffs)),
                "secondary_mean_delta": float(mean(secondary_diffs)),
                "primary_sign_test": sign_test(primary_diffs),
                "secondary_sign_test": sign_test(secondary_diffs),
            }
        )

    baseline_comparisons = [comparison for comparison in comparisons if comparison["baseline_role"] == "baseline"]
    ours_record = next(record for record in records if record["method_key"] == ours_key)
    status = {
        "task_count": len(rows),
        "baseline_count": sum(1 for _key, _label, role in METHOD_SPECS if role == "baseline"),
        "ours_primary_rank_1": ours_record["primary_rank"] == 1,
        "ours_secondary_rank_1": ours_record["secondary_rank"] == 1,
        "ours_vs_all_baselines_primary_significant_0_05": all(
            comparison["primary_sign_test"]["significant_0_05"]
            and comparison["primary_mean_delta"] > 0
            for comparison in baseline_comparisons
        ),
        "ours_vs_all_baselines_secondary_significant_0_05": all(
            comparison["secondary_sign_test"]["significant_0_05"]
            and comparison["secondary_mean_delta"] > 0
            for comparison in baseline_comparisons
        ),
        "evidence_boundary": (
            "Projection over real HarnessBench task metadata; evaluates harness capability routing, "
            "not end-to-end agent task success."
        ),
    }
    output = {
        "experiment_id": "L4-HARNESSBENCH-ROUTING-PK",
        "inputs": {"tasks_dir": str(args.tasks_dir)},
        "method_specs": [
            {"method_key": key, "method_label": label, "role": role}
            for key, label, role in METHOD_SPECS
        ],
        "status": status,
        "records": records,
        "comparisons": comparisons,
        "rows": rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "records": records, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
