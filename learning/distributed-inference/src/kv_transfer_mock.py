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
