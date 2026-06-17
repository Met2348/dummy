"""Persistent memory — file-based key/value that survives across sessions.

A real harness uses files (CLAUDE.md, a memory dir) so knowledge outlives a
single context window. Here: a JSON-backed dict. The point is *durability* and
*offload* — push facts here instead of keeping them in the window forever.
"""
from __future__ import annotations

import json
import os


class Memory:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001 — corrupt memory shouldn't crash the agent
                return {}
        return {}

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self._save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def all(self) -> dict:
        return dict(self.data)

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)


def _self_test() -> None:
    import tempfile

    d = tempfile.mkdtemp(prefix="harness_mem_")
    path = os.path.join(d, "memory.json")

    m = Memory(path)
    assert m.get("missing", "def") == "def"
    m.set("last_budget", 150)
    m.set("team", "infra")

    # Durability: a fresh Memory on the same path sees the writes.
    m2 = Memory(path)
    assert m2.get("last_budget") == 150 and m2.get("team") == "infra"
    assert set(m2.all()) == {"last_budget", "team"}

    # Corrupt file degrades to empty, doesn't raise.
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    m3 = Memory(path)
    assert m3.all() == {}

    os.remove(path)
    os.rmdir(d)
    print("[OK] harness.memory._self_test passed")


if __name__ == "__main__":
    _self_test()
