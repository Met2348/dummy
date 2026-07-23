"""06-fake-training-loop.py

配合 06-monitoring-and-debugging.md 第 2 节(ps/htop 监控自己的训练进程)使用。

只做一件事:纯 CPU 空转一段时间,制造一个"看起来像训练脚本、真的在吃 CPU"的
进程,好让 ps aux / htop 里能看到一行有意义的记录,而不是对着空空如也的进程列表
讲操作。默认跑 20 秒后自动退出,不需要手动按 Ctrl+C。

用法:
    python3 06-fake-training-loop.py [seconds]
"""
from __future__ import annotations

import sys
import time


def burn_cpu(duration_seconds: float) -> None:
    start = time.time()
    step = 0
    total = 0.0
    next_print = start + 1.0
    while time.time() - start < duration_seconds:
        # 单纯的浮点数运算,没有实际意义,只是为了让 CPU% 看起来不是 0
        total += sum(j * j for j in range(30_000))
        step += 1
        now = time.time()
        if now >= next_print:
            print(f"[fake-train] step={step} elapsed={now - start:.1f}s", flush=True)
            next_print = now + 1.0


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    print(f"[fake-train] pid={__import__('os').getpid()} starting, will burn CPU for {seconds:.0f}s", flush=True)
    burn_cpu(seconds)
    print("[fake-train] done", flush=True)
