from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import ValidationError

from traceh_core.schema import load_schema, validate_instance


SCHEMAS = Path(__file__).parents[1] / "schemas"
HASH = "a" * 64


def test_episode_schema_accepts_strict_current_record() -> None:
    schema = load_schema(SCHEMAS / "traceh-episode.schema.json")
    record = {
        "run_id": "dev-q4-task1-none-s0",
        "information_class": "DEV_SOURCE_BRANCH",
        "task_id": "task1",
        "model_id": "Qwen3-4B",
        "checkpoint_hash": HASH,
        "precision": "4bit",
        "policy_id": "NONE",
        "seed": 0,
        "success": False,
        "utility": -0.1,
        "steps": 10,
        "input_tokens": 100,
        "output_tokens": 50,
        "invalid_actions": 1,
        "wall_time_seconds": 2.5,
        "infrastructure_failure": False,
        "parser_failure": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": "1234567",
        "config_hash": HASH,
    }
    validate_instance(record, schema)
    with pytest.raises(ValidationError):
        validate_instance({**record, "unexpected": True}, schema)


def test_branch_schema_requires_matching_hash_shape() -> None:
    schema = load_schema(SCHEMAS / "traceh-branch.schema.json")
    record = {
        "run_id": "branch-001",
        "baseline_run_id": "base-001",
        "task_id": "task1",
        "model_id": "Qwen3-4B",
        "prefix_id": "prefix-0",
        "prefix_state_hash": HASH,
        "replayed_state_hash": HASH,
        "event_type": "no_progress",
        "action": "CHECK",
        "seed": 0,
        "terminal_utility": 1.0,
        "none_utility": 0.0,
        "action_advantage": 1.0,
        "steps": 4,
        "input_tokens": 40,
        "output_tokens": 20,
    }
    validate_instance(record, schema)


def test_ledger_forbids_pretest_target_action_outcomes() -> None:
    schema = load_schema(SCHEMAS / "traceh-information-ledger.schema.json")
    entry = {
        "method_id": "trace-h",
        "source_action_outcomes_seen": 100,
        "target_baseline_episodes_seen": 6,
        "target_action_outcomes_seen_before_test": 0,
        "target_action_outcomes_seen_online": 0,
        "extra_test_time_calls": 0,
        "tokens": 0,
        "steps": 0,
        "wall_clock_seconds": 0.0,
    }
    validate_instance(entry, schema)
    with pytest.raises(ValidationError):
        validate_instance({**entry, "target_action_outcomes_seen_before_test": 1}, schema)


def test_router_and_freeze_schemas() -> None:
    router_schema = load_schema(SCHEMAS / "traceh-router-artifact.schema.json")
    freeze_schema = load_schema(SCHEMAS / "traceh-freeze.schema.json")
    router = {
        "router_id": "traceh-dev",
        "version": "0.1.0",
        "feature_schema_hash": HASH,
        "source_branch_bank_hash": HASH,
        "embedding_model": "Qwen3-Embedding-0.6B",
        "transport": {
            "kind": "unbalanced_partial",
            "reg": 0.05,
            "reg_m": 0.1,
            "max_match_cost": 0.35,
        },
        "coverage_threshold": 0.5,
        "lcb_z": 1.0,
        "action_costs": {"NONE": 0, "CHECK": 0.01, "RETRY": 0.02, "REPLAN": 0.03},
        "fallback_action": "NONE",
    }
    validate_instance(router, router_schema)
    freeze = {
        "freeze_id": "dev-freeze-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": "1234567",
        "environment_lock_hash": HASH,
        "model_manifest_hash": HASH,
        "task_manifest_hash": HASH,
        "action_contract_hash": HASH,
        "source_branch_bank_hash": HASH,
        "target_baseline_hash": HASH,
        "router_artifact_hash": HASH,
        "baseline_artifact_hashes": {"source-aw": HASH},
        "statistics_script_hash": HASH,
        "target_action_outcomes_seen_before_freeze": 0,
    }
    validate_instance(freeze, freeze_schema)

