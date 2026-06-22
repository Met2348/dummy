"""Run ERIC-3080Ti local environment and test audits.

This script is intentionally small and repo-local. It runs each learning module
in isolation so import path quirks in one topic do not poison another topic.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT / "learning"
OUT_DIR = ROOT / "docs" / "local-env"
DEFAULT_JSON = OUT_DIR / "ERIC-3080Ti-test-results.json"
DEFAULT_MD = OUT_DIR / "ERIC-3080Ti-test-matrix.md"


TIERS = {
    "A": [
        "agent-code-eval",
        "agent-foundations",
        "agent-framework-stack",
        "agent-graduation",
        "agent-memory-context",
        "eval-foundations",
        "eval-graduation",
        "llm-judge-arena",
        "multi-agent-orchestration",
        "rag-essential",
        "reasoning-eval",
        "red-team-jailbreak",
        "safety-defense",
        "tool-use-mcp",
    ],
    "B": [
        "prompt-tuning-family",
        "lora-family",
        "adapter-tuning-family",
        "data-curation",
        "transformer-deep",
        "moe-architecture",
        "ssm-hybrid",
        "long-context",
        "pretraining-recipe",
        "small-model-graduation",
        "dpo-family",
        "process-reward",
        "rl-foundations",
        "rlhf-classic",
        "scaling-infra",
        "speculative-decoding",
        "production-serving",
        "serving-graduation",
    ],
    "C": [
        "inference-engine-core",
        "reasoning-r1",
        "rl-sota-2026",
        "distributed-inference",
        "quantization-deploy",
        "sglang-radixattention",
        "multimodal-agent",
    ],
    "M8": [
        "gpu-architecture",
        "cuda-essentials",
        "kernel-engineering",
        "cluster-networking",
        "storage-dataops",
        "training-orchestration",
        "infra-graduation",
    ],
}


@dataclass
class RunResult:
    module: str
    kind: str
    command: list[str]
    status: str
    returncode: int | None
    seconds: float
    stdout: str
    stderr: str


def _module_tier(module: str) -> str:
    for tier, modules in TIERS.items():
        if module in modules:
            return tier
    return "?"


def _env_for(module: str) -> dict[str, str]:
    env = os.environ.copy()
    src = str((LEARNING / module / "src").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return env


def _run(module: str, kind: str, command: list[str], timeout: int) -> RunResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=_env_for(module),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        return RunResult(
            module=module,
            kind=kind,
            command=command,
            status=status,
            returncode=proc.returncode,
            seconds=round(time.perf_counter() - start, 2),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            module=module,
            kind=kind,
            command=command,
            status="TIMEOUT",
            returncode=None,
            seconds=round(time.perf_counter() - start, 2),
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )


def _commands_for(module: str, include_env: bool, include_tests: bool) -> list[tuple[str, list[str]]]:
    base = LEARNING / module
    commands: list[tuple[str, list[str]]] = []
    verify = base / "environment" / "verify_env.py"
    tests = base / "src" / "tests"
    if include_env and verify.exists():
        commands.append(("env", [sys.executable, str(verify)]))
    if include_tests and tests.exists():
        commands.append(("tests", [sys.executable, "-m", "pytest", str(tests), "-q"]))
    return commands


def _run_test_command(module: str, command: list[str], timeout: int) -> list[RunResult]:
    """Run pytest, falling back to script-style test files when nothing is collected."""
    pytest_result = _run(module, "tests-pytest", command, timeout)
    combined = (pytest_result.stdout + "\n" + pytest_result.stderr).lower()
    if pytest_result.returncode != 5 and "no tests ran" not in combined:
        return [pytest_result]

    tests_dir = LEARNING / module / "src" / "tests"
    script_results: list[RunResult] = []
    for test_file in sorted(tests_dir.glob("test*.py")):
        script_results.append(
            _run(
                module,
                f"tests-script:{test_file.name}",
                [sys.executable, str(test_file)],
                timeout,
            )
        )
    if not script_results:
        return [pytest_result]
    return script_results


RUNBOOK_NAME = "runbook.yaml"


def _load_runbook(module: str) -> list[dict]:
    """Read learning/<module>/runbook.yaml -> list of command entries."""
    path = LEARNING / module / RUNBOOK_NAME
    if not path.exists():
        return []
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("commands", []) or []


def _format_cmd(template: str, params: dict | None) -> list[str]:
    """Render a doc command template with smoke params and swap python -> venv exe."""
    text = template.format(**(params or {}))
    tokens = [t.strip('"').strip("'") for t in shlex.split(text, posix=False)]
    if tokens and tokens[0] in {"python", "python3", "py"}:
        tokens[0] = sys.executable
    return tokens


def _script_of(tokens: list[str]) -> str | None:
    for token in tokens:
        if token.endswith(".py"):
            return token
    return None


def _run_runbook(module: str, timeout: int) -> list[RunResult]:
    """Verify a module's documented entry-point commands.

    V0: referenced script exists + `python <script> --help` exits 0.
    V1: run the smoke form of `tier: V1` commands to completion.
    """
    results: list[RunResult] = []
    for entry in _load_runbook(module):
        cid = entry.get("id", "?")
        tier = entry.get("tier", "V1")
        tokens = _format_cmd(entry["cmd"], entry.get("smoke", {}))
        script = _script_of(tokens)
        # V0 --help probe; skip for argparse-less scripts via `v0: false`.
        if script is not None and entry.get("v0", True):
            exists = (ROOT / script).exists() or Path(script).exists()
            if not exists:
                results.append(
                    RunResult(module, f"runbook-v0:{cid}", tokens, "FAIL", None,
                              0.0, "", f"script not found: {script}")
                )
            else:
                results.append(
                    _run(module, f"runbook-v0:{cid}",
                         [sys.executable, script, "--help"], min(timeout, 120))
                )
        if tier == "V1":
            results.append(_run(module, f"runbook-v1:{cid}", tokens, timeout))
    return results


def _selected_modules(args: argparse.Namespace) -> list[str]:
    if args.modules:
        return args.modules
    modules: list[str] = []
    for tier in args.tiers:
        modules.extend(TIERS[tier])
    return modules


def _write_json(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine_label": "ERIC-3080Ti",
        "python": sys.executable,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _short_reason(result: RunResult) -> str:
    if result.status == "PASS":
        return ""
    text = (result.stderr + "\n" + result.stdout).strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:180].replace("|", "\\|")
    return ""


def _write_md(results: list[RunResult], path: Path) -> None:
    lines = [
        "# ERIC-3080Ti Test Matrix",
        "",
        f"> Generated by `{Path(__file__).as_posix()}`",
        "",
        "| Tier | Module | Kind | Status | Seconds | Reason |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {_module_tier(r.module)} | `{r.module}` | {r.kind} | {r.status} | "
            f"{r.seconds:.2f} | {_short_reason(r)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIERS), default=["A"])
    parser.add_argument("--modules", nargs="*", default=[])
    parser.add_argument("--env", action="store_true", help="run verify_env.py")
    parser.add_argument("--tests", action="store_true", help="run pytest")
    parser.add_argument("--runbook", action="store_true",
                        help="run documented runbook.yaml entry-point commands (V0+V1)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    if not args.env and not args.tests and not args.runbook:
        args.tests = True

    results: list[RunResult] = []
    for module in _selected_modules(args):
        if args.runbook:
            for result in _run_runbook(module, args.timeout):
                print(f"  -> {result.kind} {result.status} ({result.seconds:.2f}s)", flush=True)
                results.append(result)
                _write_json(results, args.json_out)
                _write_md(results, args.md_out)
            if not args.env and not args.tests:
                continue
        commands = _commands_for(module, args.env, args.tests)
        if not commands:
            results.append(
                RunResult(
                    module=module,
                    kind="none",
                    command=[],
                    status="SKIP",
                    returncode=0,
                    seconds=0.0,
                    stdout="No selected audit command exists for this module.",
                    stderr="",
                )
            )
            continue
        for kind, command in commands:
            print(f"[{module}] {kind}: {' '.join(command)}", flush=True)
            if kind == "tests":
                module_results = _run_test_command(module, command, args.timeout)
            else:
                module_results = [_run(module, kind, command, args.timeout)]
            for result in module_results:
                print(f"  -> {result.kind} {result.status} ({result.seconds:.2f}s)", flush=True)
                results.append(result)
                _write_json(results, args.json_out)
                _write_md(results, args.md_out)

    _write_json(results, args.json_out)
    _write_md(results, args.md_out)
    failures = [r for r in results if r.status not in {"PASS", "SKIP"}]
    print(f"\nWrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    print(f"Summary: {len(results) - len(failures)}/{len(results)} pass/skip")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
