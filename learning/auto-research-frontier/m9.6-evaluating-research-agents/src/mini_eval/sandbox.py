"""一个最小的安全执行沙箱——真 exec 候选代码，但限制能力。

沿用你验证过的 agent-code-eval 的 `safe_exec` 哲学：**白名单 __builtins__ + 禁用模式黑名单**。
评测的地基必须是"真把候选跑起来"，否则一切判分都是假的（这正是 9.6 的立身之本）。
"""
from __future__ import annotations

import builtins as _builtins
import io
from contextlib import redirect_stdout

# 候选代码允许用的 builtins（白名单——没列的都用不了）
SAFE_BUILTINS = (
    "len", "range", "sum", "min", "max", "abs", "round", "float", "int", "bool",
    "str", "dict", "list", "tuple", "set", "enumerate", "sorted", "zip", "map",
    "filter", "print", "any", "all", "isinstance",
)

# 禁用子串（黑名单——出现即拒绝执行，防越权/逃逸）
FORBIDDEN = (
    "import", "__", "open(", "eval(", "exec(", "compile(", "globals", "locals",
    "getattr", "setattr", "delattr", "os.", "sys.", "subprocess", "socket", "input(",
)


class SafeExecError(Exception):
    pass


def safe_exec(src: str):
    """真执行 src，返回 (命名空间, 捕获的 stdout)。违规则抛 SafeExecError。"""
    for pat in FORBIDDEN:
        if pat in src:
            raise SafeExecError(f"forbidden pattern: {pat!r}")
    safe_builtins = {name: getattr(_builtins, name) for name in SAFE_BUILTINS}
    ns = {"__builtins__": safe_builtins}
    buf = io.StringIO()
    try:
        code = compile(src, "<candidate>", "exec")
        with redirect_stdout(buf):
            exec(code, ns)            # ← 真执行（受限命名空间）
    except SafeExecError:
        raise
    except Exception as e:            # 候选自身运行期错误
        raise SafeExecError(f"runtime error: {type(e).__name__}: {e}")
    return ns, buf.getvalue()
