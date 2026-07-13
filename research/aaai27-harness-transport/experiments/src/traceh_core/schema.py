"""Strict JSON Schema validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def validate_instance(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    Draft202012Validator(schema).validate(instance)

