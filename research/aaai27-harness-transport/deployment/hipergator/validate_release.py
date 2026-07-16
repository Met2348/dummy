#!/usr/bin/env python3
"""Fail fast on GitHub-to-HiPerGator portability regressions."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / "experiments" / "pyproject.toml",
    ROOT / "experiments" / "uv.lock",
    ROOT / "experiments" / "scripts" / "run_cluster_block.py",
    ROOT / "deployment" / "hipergator" / "bootstrap-env.sh",
    ROOT / "deployment" / "hipergator" / "prepare-assets.sh",
    ROOT / "deployment" / "hipergator" / "make_asset_manifest.py",
    ROOT / "deployment" / "hipergator" / "block-manifest.example.tsv",
    ROOT / "deployment" / "hipergator" / "slurm" / "gpu-preflight.sbatch",
    ROOT / "deployment" / "hipergator" / "slurm" / "rollout-block-array-b200.sbatch",
]
FORBIDDEN_RUNTIME_PATHS = (b"/mnt/c/", b"/home/wsl/", b"C:\\Workspace\\")


def tracked_files() -> list[Path]:
    try:
        git_root = Path(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--", "."],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file()]
    return [git_root / item.decode("utf-8") for item in output.split(b"\0") if item]


def main() -> None:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    for path in (ROOT / "deployment").rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if path.suffix in {".sh", ".sbatch", ".py"} and b"\r\n" in payload:
            errors.append(f"CRLF is not allowed in cluster executable: {path.relative_to(ROOT)}")
        if path.resolve() != Path(__file__).resolve() and path.suffix in {
            ".sh",
            ".sbatch",
            ".py",
            ".json",
            ".tsv",
        }:
            for marker in FORBIDDEN_RUNTIME_PATHS:
                if marker in payload:
                    errors.append(
                        f"local absolute path {marker!r} in runtime file: {path.relative_to(ROOT)}"
                    )

    manifest = ROOT / "deployment" / "hipergator" / "block-manifest.example.tsv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = set(reader.fieldnames or [])
        required_columns = {
            "block_id",
            "phase",
            "entrypoint",
            "job_spec",
            "information_class",
            "checkpoint_hash",
            "task_hash",
            "prompt_hash",
            "freeze_hash",
        }
        missing = sorted(required_columns - columns)
        if missing:
            errors.append(f"example manifest missing columns: {', '.join(missing)}")

    large_files = []
    for path in tracked_files():
        if path.is_file() and path.stat().st_size > 25 * 1024 * 1024:
            try:
                relative = path.relative_to(ROOT)
            except ValueError:
                continue
            large_files.append(str(relative))
    if large_files:
        errors.append("tracked files larger than 25 MiB: " + ", ".join(sorted(large_files)))

    report = {"ok": not errors, "project_root": str(ROOT), "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
