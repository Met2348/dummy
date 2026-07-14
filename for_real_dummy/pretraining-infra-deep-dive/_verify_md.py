import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

def extract_blocks(md_path):
    text = Path(md_path).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    # ```python-wsl2 块需要真实 Linux 语义(fork/epoll/signal等),Windows 原生 Python
    # 跑不了,单独用 wsl.exe 手动验证过并在文中记录,这里只统计数量,不尝试执行。
    wsl2_blocks = re.findall(r"```python-wsl2\n(.*?)```", text, re.DOTALL)
    return blocks, wsl2_blocks

def main():
    md_path = sys.argv[1]
    blocks, wsl2_blocks = extract_blocks(md_path)
    print(f"Found {len(blocks)} python code blocks in {md_path}")
    if wsl2_blocks:
        print(f"Found {len(wsl2_blocks)} python-wsl2 code blocks (skipped here - verify manually via wsl.exe, see in-file verification records)")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    fails = []
    # 用临时 .py 文件而不是 `-c` 字符串运行每个代码块:本系列的知识点大量用到
    # multiprocessing.Process(spawn 方式在 Windows 上需要重新 import 一个真实的
    # __main__ 模块来找到目标函数),`-c` 字符串没有对应的可导入模块文件,子进程会
    # AttributeError 崩溃且往往表现为父进程在 queue.get()/join() 上无限挂起,不是
    # 单纯的失败退出。写成临时文件跑,给 spawn 一个真实可 import 的 __main__。
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, block in enumerate(blocks, 1):
            block_path = Path(tmpdir) / f"block_{i}.py"
            block_path.write_text(block, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(block_path)],
                capture_output=True, text=True, timeout=600, env=env,
                encoding="utf-8", errors="replace",
            )
            if result.returncode != 0:
                fails.append((i, block, result.stderr))
                print(f"  [FAIL] block {i}")
            else:
                print(f"  [ok]   block {i}")
    print()
    if fails:
        print(f"{len(fails)}/{len(blocks)} blocks FAILED:")
        for i, block, err in fails:
            print(f"--- block {i} ---")
            print(block)
            print("--- stderr ---")
            print(err[-3000:])
        sys.exit(1)
    else:
        print(f"ALL {len(blocks)} blocks passed independently.")

if __name__ == "__main__":
    main()
