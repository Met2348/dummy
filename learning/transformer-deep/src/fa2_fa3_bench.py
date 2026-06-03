"""FA2 / FA3 benchmark — 数字对照 + 论文报数.

不真跑 FA3 (需 H100)，只汇总参考数字。
真跑 PyTorch SDPA + flash-attn (若装) 做相对比较。
"""
from __future__ import annotations

import time

import torch


# 论文报数（A100 baseline 100）
REFERENCE_NUMBERS = {
    "vanilla (4k)":      100,
    "FA1 (4k)":          250,
    "FA2 (4k)":          400,
    "FA2 (32k)":         900,
    "FA3 BF16 (32k, H100)":  1100,
    "FA3 FP8 (32k, H100)":   1500,
}


def real_benchmark():
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过实测")
        return
    import torch.nn.functional as F
    b, h, t, d = 2, 8, 2048, 64
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    # warmup
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    flops = 4 * b * h * t * t * d
    tflops = flops / dt / 1e12
    print(f"PyTorch SDPA on this GPU: {dt*1000:.2f} ms  ≈ {tflops:.1f} TFLOPS")


def main():
    print("\n=== 论文报数（A100 baseline 100）===")
    for name, score in REFERENCE_NUMBERS.items():
        print(f"  {name:<28} {score}")
    print("\n=== 本机实测 ===")
    real_benchmark()


if __name__ == "__main__":
    main()
