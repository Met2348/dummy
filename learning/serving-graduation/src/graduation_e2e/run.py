"""Entry point.

两种跑法都支持：
- 包形式：`python -m graduation_e2e.run --out report/`（从 src/ 目录）
- 脚本形式：`python learning/serving-graduation/src/graduation_e2e/run.py`（从 repo 根）
"""
from __future__ import annotations

import argparse
import tempfile

try:
    from .report import write_report           # `python -m graduation_e2e.run`
except ImportError:                            # 直接当脚本跑（无包上下文）
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from graduation_e2e.report import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory; if omitted, uses a tempdir")
    args = ap.parse_args()
    out_dir = args.out or tempfile.mkdtemp(prefix="grad_e2e_")
    paths = write_report(out_dir)
    print(f"Wrote {len(paths)} files to {out_dir}")
    for name in paths:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
