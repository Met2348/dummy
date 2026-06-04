"""Entry point — `python -m graduation_e2e.run --out report/`."""
from __future__ import annotations

import argparse
import tempfile

from .report import write_report


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
