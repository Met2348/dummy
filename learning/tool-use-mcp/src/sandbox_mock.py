"""Sandbox mock — AST whitelist + restricted exec."""
from __future__ import annotations
import ast


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "len": len, "max": max, "min": min,
    "range": range, "sum": sum, "sorted": sorted, "enumerate": enumerate,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "int": int, "float": float, "str": str, "bool": bool, "round": round,
    "print": None,  # filled in per-run
}

FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open",
    "globals", "locals", "vars", "__builtins__",
    "getattr", "setattr", "delattr",
}


def _has_forbidden(tree: ast.AST) -> tuple[bool, str]:
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return True, f"forbidden node: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return True, f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True, f"dunder access: {node.attr}"
    return False, ""


def run_sandboxed(code: str, timeout_ms: int = 1000) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"syntax: {e}"}

    blocked, reason = _has_forbidden(tree)
    if blocked:
        return {"ok": False, "error": reason}

    out: list[str] = []
    def safe_print(*args, **kwargs):  # noqa: ARG001
        out.append(" ".join(str(a) for a in args))

    builtins = dict(SAFE_BUILTINS)
    builtins["print"] = safe_print

    ns: dict = {"__builtins__": builtins}
    try:
        exec(compile(tree, "<sandbox>", "exec"), ns, ns)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "stdout": "\n".join(out)}
    return {"ok": True, "stdout": "\n".join(out)}


def _self_test() -> None:
    r = run_sandboxed("print(1+2)")
    assert r["ok"] and r["stdout"] == "3", r

    r = run_sandboxed("print(sum(range(5)))")
    assert r["ok"] and r["stdout"] == "10", r

    r = run_sandboxed("import os")
    assert not r["ok"] and "forbidden" in r["error"], r

    r = run_sandboxed("eval('1+1')")
    assert not r["ok"] and "forbidden" in r["error"], r

    r = run_sandboxed("[].__class__.__bases__")
    assert not r["ok"] and "dunder" in r["error"], r

    r = run_sandboxed("x = 1/0")
    assert not r["ok"] and "division" in r["error"].lower(), r
    print("[OK] sandbox_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
