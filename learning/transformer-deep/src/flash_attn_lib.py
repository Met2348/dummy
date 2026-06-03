"""flash-attn 库调用 — 与 naive 对照 benchmark."""
from __future__ import annotations

import time

import torch


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


def bench_pytorch_sdpa(q, k, v, n_iter: int = 10):
    import torch.nn.functional as F
    torch.cuda.synchronize() if q.is_cuda else None
    t0 = time.time()
    for _ in range(n_iter):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize() if q.is_cuda else None
    return out, (time.time() - t0) / n_iter


def bench_flash_attn(q, k, v, n_iter: int = 10):
    from flash_attn import flash_attn_func
    # flash_attn 期望 (b, t, h, d) 排列
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()
    torch.cuda.synchronize() if q.is_cuda else None
    t0 = time.time()
    for _ in range(n_iter):
        out = flash_attn_func(q1, k1, v1, causal=True)
    torch.cuda.synchronize() if q.is_cuda else None
    return out.transpose(1, 2), (time.time() - t0) / n_iter


def run_demo() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    b, h, t, d = 2, 8, 1024, 64
    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype)
    print(f"device={device} dtype={dtype} shape=({b},{h},{t},{d})")

    out_sdpa, t_sdpa = bench_pytorch_sdpa(q, k, v)
    print(f"  PyTorch SDPA:  {t_sdpa*1000:.2f} ms")

    if has_flash_attn() and device == "cuda":
        out_fa, t_fa = bench_flash_attn(q, k, v)
        print(f"  flash-attn:    {t_fa*1000:.2f} ms")
        diff = (out_sdpa - out_fa).abs().max().item()
        print(f"  SDPA vs FA max diff: {diff:.2e}")
    else:
        print("  [SKIP] flash-attn 不可用 (Win/CPU)")


if __name__ == "__main__":
    run_demo()
