"""
harness-engineering 专题环境自检.
运行: python environment/verify_env.py
通过标准: stdlib 组件可用, src 生产级组件可 import, MockProvider 可产出确定性流式.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 核心运行只依赖 stdlib; 这些是 notebook 可视化/生成所需
OPTIONAL = ["nbformat", "pandas", "matplotlib"]
# 本专题 src 生产级组件
SRC_MODS = ["provider", "compaction", "long_horizon", "otel_trace", "harness_eval"]


def check_optional() -> None:
    print("== Part A: notebook 依赖 (可视化/生成) ==")
    for pkg in OPTIONAL:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}{'' if ok else '  (notebook 可视化需要)'}")


def check_src() -> bool:
    print("== Part B: src 生产级组件 import ==")
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    ok = True
    for m in SRC_MODS:
        present = importlib.util.find_spec(m) is not None
        print(f"  [{'OK ' if present else 'MISS'}] src/{m}.py")
        ok = ok and present
    return ok


def check_mock_provider() -> bool:
    print("== Part C: MockProvider 确定性流式 smoke ==")
    try:
        import provider as P
        prov = P.MockProvider()
        msgs = [{"role": "user", "content": "请用 echo 工具复述 hello"}]
        chunks = list(prov.stream(msgs, tools=[{"name": "echo"}]))
        text = "".join(c.text for c in chunks if c.kind == "text")
        has_tool = any(c.kind == "tool_call" for c in chunks)
        print(f"  [OK ] 产出 {len(chunks)} 个 chunk, 含文本={bool(text)}, 含 tool_call={has_tool}")
        return len(chunks) > 0
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def main() -> int:
    check_optional()
    src_ok = check_src()
    mock_ok = check_mock_provider()
    print("\n== 结论 ==")
    if src_ok and mock_ok:
        print("核心组件全部通过 ✅  notebook 可运行 (默认 MockProvider, 无需 API key).")
        return 0
    print("有组件未通过, 请检查上面 MISS/FAIL 项.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
