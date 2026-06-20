"""Unit tests for the --runbook mode helpers in eric_3080ti_env_audit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eric_3080ti_env_audit as audit  # noqa: E402


def test_format_cmd_substitutes_and_swaps_python():
    tokens = audit._format_cmd("python src/x.py --steps {steps}", {"steps": 2000})
    assert tokens[0] == sys.executable
    assert tokens[-2:] == ["--steps", "2000"]


def test_format_cmd_no_params_is_literal():
    tokens = audit._format_cmd("python src/x.py --flag", {})
    assert tokens[1:] == ["src/x.py", "--flag"]


def test_script_of_finds_py():
    assert audit._script_of([sys.executable, "src/x.py", "--a"]) == "src/x.py"
    assert audit._script_of([sys.executable, "-m", "pytest"]) is None


def test_load_runbook_reads_commands(tmp_path, monkeypatch):
    mod = tmp_path / "learning" / "demo"
    mod.mkdir(parents=True)
    (mod / "runbook.yaml").write_text(
        "module: demo\ncommands:\n  - id: a\n    cmd: 'python src/x.py'\n    tier: V0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "LEARNING", tmp_path / "learning")
    cmds = audit._load_runbook("demo")
    assert cmds[0]["id"] == "a"


def test_load_runbook_missing_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LEARNING", tmp_path / "learning")
    assert audit._load_runbook("nope") == []
