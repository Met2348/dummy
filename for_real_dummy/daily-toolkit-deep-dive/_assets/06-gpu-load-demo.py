"""06-gpu-load-demo.py

配合 06-monitoring-and-debugging.md 第 1 节(nvidia-smi 日常怎么用)使用。

只做一件事:在 GPU 上反复做大矩阵乘法,制造一段持续几十秒、真实能在 nvidia-smi
里看到 GPU-Util 和显存占用的负载——不是真实训练,只是"占位负载",目的是让你
看 nvidia-smi 输出的时候,屏幕上有真实数字在变化,而不是对着一台空闲 GPU 讲操作。

用法(在仓库根目录 .venv 下运行):
    .venv/Scripts/python.exe for_real_dummy/daily-toolkit-deep-dive/_assets/06-gpu-load-demo.py [seconds]

默认跑 60 秒后自动退出,不需要手动 Ctrl+C;也可以在另一个终端用
`taskkill /F /PID <pid>` 中途杀掉它,配合文档里"显存不释放"那一节演示杀进程前后的对比。
"""
from __future__ import annotations

import sys
import time

import torch


def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    assert torch.cuda.is_available(), "torch 看不到 CUDA 设备,先检查驱动/torch 安装是否正常"

    device = torch.device("cuda:0")
    print(f"[06-gpu-load-demo] pid={__import__('os').getpid()} device={torch.cuda.get_device_name(0)}", flush=True)

    # 4096x4096 float32 矩阵,单块约 64MB,两块 + 输出块,占用几百 MB 显存,
    # 足够在 nvidia-smi 的 Memory-Usage 列里看出明显变化。
    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)

    start = time.time()
    step = 0
    next_print = start + 2.0
    while time.time() - start < duration:
        c = a @ b  # 矩阵乘法,吃 GPU 算力,让 GPU-Util 不是 0%
        torch.cuda.synchronize()
        step += 1
        now = time.time()
        if now >= next_print:
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f"[06-gpu-load-demo] step={step} elapsed={now - start:.1f}s allocated={allocated:.0f}MiB", flush=True)
            next_print = now + 2.0

    print(f"[06-gpu-load-demo] done, total steps={step}", flush=True)


if __name__ == "__main__":
    main()
