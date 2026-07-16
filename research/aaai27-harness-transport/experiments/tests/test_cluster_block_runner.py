from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_cluster_block.py"
SPEC = importlib.util.spec_from_file_location("run_cluster_block", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(**overrides: str) -> dict[str, str]:
    value = {
        "block_id": "cal-q4-000",
        "phase": "target_baseline",
        "entrypoint": "qwen_alfworld_baseline",
        "model_id": "Qwen3-4B",
        "model_path": "/blue/group/user/models/Qwen3-4B",
        "precision": "bf16",
        "job_spec": "spec.json",
        "seed": "20260716",
        "information_class": "TARGET_BASELINE_CALIBRATION",
        "policy_id": "NONE",
        "checkpoint_hash": "AUTO",
        "task_hash": "AUTO",
        "prompt_hash": "AUTO",
        "action_budget": "50",
        "freeze_hash": "NA",
    }
    value.update(overrides)
    return value


def test_manifest_requires_unique_block_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "blocks.tsv"
    columns = sorted(MODULE.REQUIRED_COLUMNS)
    line = "\t".join(row()[column] for column in columns)
    manifest.write_text("\t".join(columns) + "\n" + line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        MODULE.read_manifest(manifest)


def test_target_final_requires_hashes_and_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    final = row(
        phase="target_final",
        information_class="TARGET_FINAL_TEST",
        freeze_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint_hash"):
        MODULE.validate_row(final)
    final.update(
        checkpoint_hash="b" * 64,
        task_hash="c" * 64,
        prompt_hash="d" * 64,
    )
    with pytest.raises(ValueError, match="TRACEH_FROZEN_POLICY_SHA256"):
        MODULE.validate_row(final)
    monkeypatch.setenv("TRACEH_FROZEN_POLICY_SHA256", "a" * 64)
    MODULE.validate_row(final)


def test_build_command_is_allowlisted_and_uses_block_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    script = project / "experiments" / "scripts" / "run_qwen_alfworld_baseline.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test\n", encoding="utf-8")
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    checkpoint = tmp_path / "checkpoint.sha256"
    config = tmp_path / "config.yaml"
    checkpoint.write_text("hashes\n", encoding="utf-8")
    config.write_text("env: {}\n", encoding="utf-8")
    current = row(model_path=str(model))
    spec = {
        "checkpoint_manifest": str(checkpoint),
        "config": str(config),
        "data": str(data),
        "run_label": "cluster-smoke",
        "episodes": 30,
        "offset_start": 60,
    }
    output = tmp_path / "output"
    command = MODULE.build_command(
        current,
        spec,
        project_root=project,
        output_dir=output,
        python_bin="python",
    )
    assert command[0] == "python"
    assert str(output / "raw") in command
    assert str(output / "report.json") in command
    assert command[command.index("--offset-start") + 1] == "60"
    assert command[command.index("--precision") + 1] == "bf16"
    assert command[command.index("--information-class") + 1] == "TARGET_BASELINE_CALIBRATION"


def test_job_spec_rejects_arbitrary_arguments() -> None:
    spec = {
        "checkpoint_manifest": "x",
        "config": "x",
        "data": "x",
        "run_label": "x",
        "episodes": 1,
        "shell_command": "rm -rf /",
    }
    with pytest.raises(ValueError, match="unsupported keys"):
        MODULE.require_keys(
            spec,
            {"checkpoint_manifest", "config", "data", "run_label", "episodes"},
            set(),
        )


def test_frozen_artifact_hash_must_match_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    task_manifest = tmp_path / "tasks.json"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    task_manifest.write_text("tasks\n", encoding="utf-8")
    current = row(
        checkpoint_hash=MODULE.sha256_file(checkpoint),
        task_hash="0" * 64,
    )
    spec = {
        "checkpoint_manifest": str(checkpoint),
        "task_manifest": str(task_manifest),
    }
    with pytest.raises(ValueError, match="task_manifest SHA-256"):
        MODULE.verify_declared_artifacts(current, spec, project_root=tmp_path)
