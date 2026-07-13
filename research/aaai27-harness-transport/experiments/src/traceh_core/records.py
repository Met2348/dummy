"""Crash-aware append-only record storage.

Each complete record is an immutable JSON file. A temporary file is fsynced and
then hard-linked into the records directory, so duplicate run IDs fail instead
of overwriting prior outcomes. Temporary files are never read as observations.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")


class DuplicateRunIdError(RuntimeError):
    pass


class InvalidRunIdError(ValueError):
    pass


class AppendOnlyRecordStore:
    def __init__(
        self,
        root: Path,
        validator: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.tmp_dir = self.root / ".tmp"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.validator = validator

    def append(self, record: Mapping[str, Any]) -> Path:
        run_id = str(record.get("run_id", ""))
        if not RUN_ID_RE.fullmatch(run_id):
            raise InvalidRunIdError(f"unsafe or empty run_id: {run_id!r}")
        if self.validator is not None:
            self.validator(record)

        final_path = self.records_dir / f"{run_id}.json"
        if final_path.exists():
            raise DuplicateRunIdError(run_id)

        tmp_path = self.tmp_dir / f"{run_id}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with tmp_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp_path, final_path)
            except FileExistsError as exc:
                raise DuplicateRunIdError(run_id) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return final_path

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.records_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if self.validator is not None:
                self.validator(record)
            yield record

    def count(self) -> int:
        return sum(1 for _ in self.records_dir.glob("*.json"))

