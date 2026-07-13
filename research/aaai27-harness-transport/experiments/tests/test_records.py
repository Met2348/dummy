from __future__ import annotations

import json

import pytest

from traceh_core.records import AppendOnlyRecordStore, DuplicateRunIdError, InvalidRunIdError


def test_append_only_store_rejects_duplicate_run_ids(tmp_path) -> None:
    store = AppendOnlyRecordStore(tmp_path)
    store.append({"run_id": "run-001", "value": 1})
    with pytest.raises(DuplicateRunIdError):
        store.append({"run_id": "run-001", "value": 2})
    assert list(store.iter_records()) == [{"run_id": "run-001", "value": 1}]


def test_partial_tmp_file_is_never_visible(tmp_path) -> None:
    store = AppendOnlyRecordStore(tmp_path)
    (store.tmp_dir / "interrupted.tmp").write_text('{"run_id":"bad"', encoding="utf-8")
    store.append({"run_id": "run-002", "value": 2})
    assert store.count() == 1
    assert [row["run_id"] for row in store.iter_records()] == ["run-002"]


def test_run_id_cannot_escape_record_root(tmp_path) -> None:
    store = AppendOnlyRecordStore(tmp_path)
    with pytest.raises(InvalidRunIdError):
        store.append({"run_id": "../escape", "value": 1})


def test_record_is_canonical_json(tmp_path) -> None:
    store = AppendOnlyRecordStore(tmp_path)
    path = store.append({"value": {"z": 2, "a": 1}, "run_id": "canonical"})
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["value"] == {"a": 1, "z": 2}

