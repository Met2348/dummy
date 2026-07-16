from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "deployment"
    / "hipergator"
    / "make_asset_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("make_asset_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_asset_manifest_is_deterministic_and_ignores_cache(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    (root / "weights").mkdir(parents=True)
    (root / ".cache").mkdir()
    (root / "weights" / "b.bin").write_bytes(b"b")
    (root / "a.json").write_bytes(b"a")
    (root / ".cache" / "download.tmp").write_bytes(b"ignored")

    first = MODULE.build_manifest(root, asset_id="demo", source="local", revision="r1")
    second = MODULE.build_manifest(root, asset_id="demo", source="local", revision="r1")

    assert first == second
    assert [item["path"] for item in first["files"]] == ["a.json", "weights/b.bin"]
    assert first["total_bytes"] == 2
