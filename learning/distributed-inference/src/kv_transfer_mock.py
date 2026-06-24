"""KV transfer cost helpers - bandwidth-only model."""
from __future__ import annotations

from dataclasses import dataclass


BANDWIDTHS_GBPS = {
    "nvlink_4": 900.0,
    "pcie_5": 60.0,
    "ib_400g": 50.0,
    "tcp_10g": 1.25,
}


def transfer_time_ms(payload_bytes: int, link: str) -> float:
    bw = BANDWIDTHS_GBPS[link]
    return payload_bytes / (bw * 1e9) * 1000.0


def kv_payload_bytes(seq_len: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2, n_layers: int = 32) -> int:
    return 2 * n_kv_heads * head_dim * dtype_bytes * n_layers * seq_len


def streaming_overlap(prefill_ms: float, transfer_ms: float, decode_ms: float) -> float:
    """Streaming: send each layer's KV as soon as it's produced.

    Effective wall-clock is about max(prefill, transfer) + decode (the second of two
    overlapping streams determines TTFT).
    """
    return max(prefill_ms, transfer_ms) + decode_ms


def batched_no_overlap(prefill_ms: float, transfer_ms: float, decode_ms: float) -> float:
    return prefill_ms + transfer_ms + decode_ms


def demo() -> None:
    """Show KV-cache transfer cost per link, and the streaming-overlap win."""
    print("=== KV-cache transfer cost (bandwidth-only model) ===")
    # 7B-class: 8 KV heads, head_dim 128, fp16, 32 layers.
    seq = 8192
    payload = kv_payload_bytes(seq, n_kv_heads=8, head_dim=128, dtype_bytes=2, n_layers=32)
    print(f"7B model, seq={seq}: KV payload = {payload / 1e9:.2f} GB\n")
    print(f"{'link':>10}{'GB/s':>9}{'transfer ms':>14}")
    for link, bw in BANDWIDTHS_GBPS.items():
        print(f"{link:>10}{bw:>9}{transfer_time_ms(payload, link):>14.1f}")
    print("=> NVLink hides it (~ms); IB is 10x slower; a 10G TCP link is hundreds of ms.")

    print("\n--- streaming per-layer KV vs ship-all-then-decode ---")
    prefill_ms, transfer_ms, decode_ms = 100.0, 80.0, 200.0
    s = streaming_overlap(prefill_ms, transfer_ms, decode_ms)
    b = batched_no_overlap(prefill_ms, transfer_ms, decode_ms)
    print(f"prefill={prefill_ms}ms transfer={transfer_ms}ms decode={decode_ms}ms")
    print(f"  batched (prefill + transfer + decode) = {b:.0f} ms")
    print(f"  streaming (max(prefill,transfer) + decode) = {s:.0f} ms")
    print(f"=> overlap saves {b - s:.0f} ms by sending each layer's KV as it is produced.")


if __name__ == "__main__":
    demo()
