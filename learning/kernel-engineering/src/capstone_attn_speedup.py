"""Capstone: naive vs FlashAttn HBM traffic for varying seq lengths."""
from __future__ import annotations


def hbm_naive_attn(N: int, d: int, dtype_bytes: int = 2) -> int:
    """Q, K, V read + S=N×N matrix written + P=N×N written + output written."""
    qkv = 3 * N * d
    s_matrix = N * N      # written and re-read
    p_matrix = N * N
    out = N * d
    return dtype_bytes * (qkv + 2 * s_matrix + 2 * p_matrix + out)


def hbm_flash_attn(N: int, d: int, dtype_bytes: int = 2) -> int:
    """Q, K, V read + output written. No N×N materialized to HBM."""
    qkv = 3 * N * d
    out = N * d
    return dtype_bytes * (qkv + out)


def speedup_curve(d: int = 128) -> list[dict]:
    rows = []
    for N in [512, 2048, 8192, 32768, 131072]:
        n_b = hbm_naive_attn(N, d)
        f_b = hbm_flash_attn(N, d)
        rows.append({
            "seq_len": N,
            "naive_mb": round(n_b / 1e6, 1),
            "flash_mb": round(f_b / 1e6, 1),
            "speedup": round(n_b / f_b, 1),
        })
    return rows


def _self_test() -> None:
    rows = speedup_curve()
    # Speedup grows with N (since N² vs N)
    speedups = [r["speedup"] for r in rows]
    assert speedups == sorted(speedups), speedups
    assert speedups[-1] > 100, speedups[-1]      # 128k seq → huge advantage
    print(f"[OK] capstone_attn_speedup (128k seq: {speedups[-1]}x HBM saved)")


if __name__ == "__main__":
    _self_test()
    print()
    print("Seq len | Naive MB | Flash MB | Speedup")
    print("--------|----------|----------|--------")
    for r in speedup_curve():
        print(f"{r['seq_len']:>7} | {r['naive_mb']:>8} | {r['flash_mb']:>8} | {r['speedup']:>5}x")
