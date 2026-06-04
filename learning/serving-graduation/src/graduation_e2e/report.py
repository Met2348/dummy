"""Graduation report — runs compare, writes markdown + JSON to a directory."""
from __future__ import annotations

import json
import os
from typing import Dict

from .compare import run_compare, to_md


def write_report(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    report = run_compare()
    md = to_md(report)
    paths = {
        "report.md": md,
        "report.json": json.dumps(report, indent=2),
    }
    for name, content in paths.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(content)
    return paths
