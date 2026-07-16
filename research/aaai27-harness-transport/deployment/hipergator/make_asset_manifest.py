#!/usr/bin/env python3
"""Create a deterministic content manifest for a model or dataset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, *, asset_id: str, source: str, revision: str) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"asset tree has no files: {root}")
    return {
        "schema_version": 1,
        "asset_id": asset_id,
        "source": source,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        args.root,
        asset_id=args.asset_id,
        source=args.source,
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manifest_sha256": sha256_file(args.output),
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
