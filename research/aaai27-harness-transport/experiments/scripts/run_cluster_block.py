#!/usr/bin/env python3
"""Execute one validated TRACE-H Slurm manifest block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "block_id",
    "phase",
    "entrypoint",
    "model_id",
    "model_path",
    "precision",
    "job_spec",
    "seed",
    "information_class",
    "policy_id",
    "checkpoint_hash",
    "task_hash",
    "prompt_hash",
    "action_budget",
    "freeze_hash",
}
PHASE_INFORMATION_CLASS = {
    "source_branch": "SOURCE_BRANCH",
    "source_validation": "SOURCE_VALIDATION",
    "target_baseline": "TARGET_BASELINE_CALIBRATION",
    "target_final": "TARGET_FINAL_TEST",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOCK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_path(value: str, *, base: Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute() and base is not None:
        expanded = base / expanded
    return expanded.resolve()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(missing)}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("manifest has no blocks")
    block_ids = [row["block_id"] for row in rows]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("manifest block_id values must be unique")
    return rows


def validate_row(row: dict[str, str]) -> None:
    if not BLOCK_ID_RE.fullmatch(row["block_id"]):
        raise ValueError(f"invalid block_id: {row['block_id']!r}")
    expected_class = PHASE_INFORMATION_CLASS.get(row["phase"])
    if expected_class is None:
        raise ValueError(f"unsupported phase: {row['phase']!r}")
    if row["information_class"] != expected_class:
        raise ValueError(
            f"phase {row['phase']!r} requires information_class {expected_class!r}"
        )
    if row["entrypoint"] != "qwen_alfworld_baseline":
        raise ValueError(f"unsupported entrypoint: {row['entrypoint']!r}")
    if row["policy_id"] != "NONE":
        raise ValueError("qwen_alfworld_baseline only supports policy_id=NONE")
    if row["precision"] not in {"bf16", "nf4"}:
        raise ValueError(f"unsupported precision: {row['precision']!r}")
    try:
        budget = int(row["action_budget"])
        int(row["seed"])
    except ValueError as error:
        raise ValueError("seed/action_budget must be integers") from error
    if budget <= 0:
        raise ValueError("action_budget must be positive")
    for field in ("checkpoint_hash", "task_hash", "prompt_hash"):
        if row[field] != "AUTO" and not SHA256_RE.fullmatch(row[field]):
            raise ValueError(f"{field} must be AUTO or a lowercase SHA-256")
    if row["phase"] == "target_final":
        for field in ("checkpoint_hash", "task_hash", "prompt_hash", "freeze_hash"):
            if not SHA256_RE.fullmatch(row[field]):
                raise ValueError(f"target_final requires a frozen SHA-256 in {field}")
        acknowledged = os.environ.get("TRACEH_FROZEN_POLICY_SHA256", "")
        if acknowledged != row["freeze_hash"]:
            raise ValueError(
                "target_final requires TRACEH_FROZEN_POLICY_SHA256 to match freeze_hash"
            )
    elif row["freeze_hash"] not in {"NA", "AUTO"} and not SHA256_RE.fullmatch(
        row["freeze_hash"]
    ):
        raise ValueError("freeze_hash must be NA/AUTO or a lowercase SHA-256")


def load_job_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("job spec must be a JSON object")
    return spec


def require_keys(spec: dict[str, Any], required: set[str], optional: set[str]) -> None:
    missing = sorted(required - set(spec))
    unknown = sorted(set(spec) - required - optional)
    if missing:
        raise ValueError(f"job spec missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"job spec has unsupported keys: {', '.join(unknown)}")


def verify_declared_artifacts(
    row: dict[str, str], spec: dict[str, Any], *, project_root: Path
) -> dict[str, dict[str, str]]:
    declarations = {
        "checkpoint_hash": "checkpoint_manifest",
        "task_hash": "task_manifest",
        "prompt_hash": "prompt_artifact",
        "freeze_hash": "freeze_artifact",
    }
    resolved: dict[str, dict[str, str]] = {}
    for hash_field, spec_key in declarations.items():
        declared = row[hash_field]
        if declared in {"AUTO", "NA"} and hash_field != "checkpoint_hash":
            continue
        if spec_key not in spec:
            raise ValueError(f"{hash_field} is frozen but job spec has no {spec_key}")
        path = expand_path(str(spec[spec_key]), base=project_root)
        if not path.is_file():
            raise FileNotFoundError(f"{spec_key} does not exist: {path}")
        actual = sha256_file(path)
        if declared != "AUTO" and actual != declared:
            raise ValueError(f"{spec_key} SHA-256 does not match {hash_field}")
        resolved[spec_key] = {"path": str(path), "sha256": actual}
    return resolved


def build_command(
    row: dict[str, str],
    spec: dict[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    python_bin: str,
) -> list[str]:
    required = {"checkpoint_manifest", "config", "data", "run_label", "episodes"}
    optional = {
        "offset_start",
        "max_steps",
        "max_input_tokens",
        "max_new_tokens",
        "action_protocol",
        "decoder",
        "task_manifest",
        "prompt_artifact",
        "freeze_artifact",
    }
    require_keys(spec, required, optional)
    checkpoint_manifest = expand_path(str(spec["checkpoint_manifest"]), base=project_root)
    config = expand_path(str(spec["config"]), base=project_root)
    data = expand_path(str(spec["data"]), base=project_root)
    model_path = expand_path(row["model_path"], base=project_root)
    for name, path in {
        "checkpoint_manifest": checkpoint_manifest,
        "config": config,
        "data": data,
        "model_path": model_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    verify_declared_artifacts(row, spec, project_root=project_root)

    script = project_root / "experiments" / "scripts" / "run_qwen_alfworld_baseline.py"
    if not script.is_file():
        raise FileNotFoundError(f"entrypoint script not found: {script}")
    command = [
        python_bin,
        str(script),
        "--project-root",
        str(project_root),
        "--model",
        str(model_path),
        "--model-id",
        row["model_id"],
        "--checkpoint-manifest",
        str(checkpoint_manifest),
        "--config",
        str(config),
        "--data",
        str(data),
        "--raw-root",
        str(output_dir / "raw"),
        "--report",
        str(output_dir / "report.json"),
        "--run-label",
        str(spec["run_label"]),
        "--episodes",
        str(spec["episodes"]),
        "--offset-start",
        str(spec.get("offset_start", 0)),
        "--max-steps",
        str(spec.get("max_steps", row["action_budget"])),
        "--max-input-tokens",
        str(spec.get("max_input_tokens", 4096)),
        "--max-new-tokens",
        str(spec.get("max_new_tokens", 32)),
        "--action-protocol",
        str(spec.get("action_protocol", "exact-text-v1")),
        "--decoder",
        str(spec.get("decoder", "command-trie-v1")),
        "--seed",
        row["seed"],
        "--precision",
        row["precision"],
        "--information-class",
        row["information_class"],
        "--policy-id",
        row["policy_id"],
    ]
    return command


def git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--", str(project_root)))
        remote = run("config", "--get", "remote.origin.url")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None, "remote": "unknown"}
    return {"commit": commit, "dirty": dirty, "remote": remote}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "alfworld"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_metadata() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    keys = ("index", "name", "uuid", "memory_total_mib", "driver_version")
    return [dict(zip(keys, (item.strip() for item in line.split(",")))) for line in output.splitlines()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve()
    rows = read_manifest(manifest)
    if args.block_index < 0 or args.block_index >= len(rows):
        raise IndexError(f"block-index {args.block_index} outside [0, {len(rows) - 1}]")
    row = rows[args.block_index]
    validate_row(row)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    project_root = expand_path(os.environ.get("TRACEH_PROJECT_ROOT", str(args.project_root)))
    job_spec_path = expand_path(row["job_spec"], base=manifest.parent)
    spec = load_job_spec(job_spec_path)
    command = build_command(
        row,
        spec,
        project_root=project_root,
        output_dir=output_dir,
        python_bin=args.python,
    )
    resolved_artifacts = verify_declared_artifacts(row, spec, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".traceh-block.lock"
    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    metadata = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dry_run": args.dry_run,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_block_index": args.block_index,
        "block": row,
        "job_spec": str(job_spec_path),
        "job_spec_sha256": sha256_file(job_spec_path),
        "resolved_artifacts": resolved_artifacts,
        "output_dir": str(output_dir),
        "command": command,
        "git": git_metadata(project_root),
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": package_versions(),
            "gpu": gpu_metadata(),
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    write_json(output_dir / "run-metadata.json", metadata)
    write_json(output_dir / "status.json", {"state": "DRY_RUN" if args.dry_run else "RUNNING", "at": utc_now()})
    if args.dry_run:
        return 0

    started_at = utc_now()
    try:
        completed = subprocess.run(command, cwd=project_root, check=False)
    except BaseException as error:
        write_json(
            output_dir / "status.json",
            {
                "state": "FAILED_TO_START",
                "started_at": started_at,
                "finished_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    state = "SUCCEEDED" if completed.returncode == 0 else "FAILED"
    write_json(
        output_dir / "status.json",
        {
            "state": state,
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
        },
    )
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=os.environ.get("TRACEH_PYTHON", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
