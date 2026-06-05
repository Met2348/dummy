# L08 · Sandbox / e2b / pyodide

## Sandbox 必要性

> LLM 生成代码 → 执行 → 拿结果 给 LLM 是 Code Interpreter 范式。
>
> 但 LLM 可能：
> - 写 `os.system("rm -rf /")` (恶意 prompt 注入)
> - 写无限循环
> - 写大内存分配
>
> Sandbox 是**强制隔离**。

## 主流 sandbox 方案

| 方案 | 隔离层 | 启动 | 适合 |
|------|--------|------|------|
| **e2b** | Firecracker VM | 200ms | 商业首选 |
| **Pyodide** | WASM | 1s | 浏览器 / Edge |
| **Modal** | Container | 1-3s | 通用 |
| **Daytona** | Container | 1s | 开源 |
| **Restricted Python** | Process | 即时 | 教学 |
| **gVisor** | User-kernel | 500ms | Google 内部 |

## e2b (起 commercial 2023, GA 2024)

```python
from e2b import Sandbox
sandbox = Sandbox()
result = sandbox.run_python("print(2 + 2)")
print(result.text)  # "4"
sandbox.close()
```

- Firecracker μVM
- Python / Node / 任意 runtime
- File system / network 可控

## Pyodide

```python
from pyodide.runners import run
result = run("import numpy as np; print(np.mean([1,2,3]))")
```

WASM-based，浏览器友好（JupyterLite 用此）。

## Restricted Python（教学）

```python
ALLOWED_BUILTINS = {
    "print": print, "len": len, "range": range,
    "abs": abs, "sum": sum, "min": min, "max": max,
    "int": int, "float": float, "str": str, "list": list, "dict": dict,
}

def safe_exec(code: str) -> str:
    out = []
    def captured_print(*args):
        out.append(" ".join(str(a) for a in args))
    locals_ = {"__builtins__": {**ALLOWED_BUILTINS, "print": captured_print}}
    exec(code, locals_, locals_)
    return "\n".join(out)
```

**警告**：Restricted Python **不是真 sandbox**！只是限制 builtins，绕过容易。教学用。

## Forbidden patterns

```python
FORBIDDEN = [
    "import os", "import sys", "__import__",
    "eval(", "exec(", "compile(",
    "open(", "globals(", "locals(",
    "__builtins__",
]
```

正则 / AST 检查，再 ⊕ restricted exec 双保险。

## 真生产架构

```
LLM → tool_call("python", code)
   ↓
[Sandbox API: e2b / Modal / Daytona]
   ↓
Run in isolated VM (no network or restricted network)
   ↓
Result + stdout + files → back to LLM
```

## 实现 (`sandbox_mock.py` 预告)

```python
import ast

class SandboxMock:
    BUILTINS_WHITELIST = {...}
    FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)

    def run(self, code: str) -> dict:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, self.FORBIDDEN_NODES):
                    return {"ok": False, "error": "import forbidden"}
            out = []
            ns = {"__builtins__": self._safe_builtins(out)}
            exec(compile(tree, "<sandbox>", "exec"), ns, ns)
            return {"ok": True, "stdout": "\n".join(out)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
```

## 退出条件

- 能列 4 sandbox 方案
- 知道 e2b 是商业首选
- 能写 AST 黑名单

## 一句话

> Sandbox = LLM 执行任意代码的安全边界 — 商业用 e2b，教学用 restricted exec + AST 黑名单。
