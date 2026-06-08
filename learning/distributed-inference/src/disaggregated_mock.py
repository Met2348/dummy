"""Disaggregated Prefill/Decode mock - single-process simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class WorkloadConfig:
    n_reqs: int = 32
    prompt_len: int = 1024
    out_len: int = 128


@dataclass
class HardwareConfig:
    prefill_ms_per_token: float = 0.3
    decode_ms_per_token: float = 8.0
    kv_bytes_per_token: int = 256_000      # 7B-class
    transfer_bw_gbps: float = 900.0        # NVLink default


def kv_transfer_ms(prompt_len: int, hw: HardwareConfig) -> float:
    payload = prompt_len * hw.kv_bytes_per_token
    return payload / (hw.transfer_bw_gbps * 1e9) * 1000.0


def colocate(w: WorkloadConfig, hw: HardwareConfig) -> dict:
    prefill = w.prompt_len * hw.prefill_ms_per_token
    decode = w.out_len * hw.decode_ms_per_token
    # serial per request, but batch=1 idealised
    per_req = prefill + decode
    total = per_req * w.n_reqs / 4         # crude batch=4 assumption
    tput = w.n_reqs * w.out_len / (total / 1000.0)
    return {"config": "colocate", "ttft_ms": round(prefill, 1), "tpot_ms": hw.decode_ms_per_token,
            "tok_per_s": round(tput, 1), "wall_s": round(total / 1000.0, 2)}


def disagg(w: WorkloadConfig, hw: HardwareConfig, cross_node: bool = False) -> dict:
    bw = 50.0 if cross_node else hw.transfer_bw_gbps
    hw2 = HardwareConfig(**{**hw.__dict__, "transfer_bw_gbps": bw})
    prefill = w.prompt_len * hw.prefill_ms_per_token
    transfer = kv_transfer_ms(w.prompt_len, hw2)
    decode = w.out_len * hw.decode_ms_per_token
    ttft = prefill + transfer
    # Prefill GPUs can pipeline the next request while decode GPUs run old ones.
    total = (prefill + transfer + decode) * w.n_reqs / 8
    tput = w.n_reqs * w.out_len / (total / 1000.0)
    cfg = "disagg-remote" if cross_node else "disagg-near"
    return {"config": cfg, "ttft_ms": round(ttft, 1), "tpot_ms": hw.decode_ms_per_token,
            "tok_per_s": round(tput, 1), "wall_s": round(total / 1000.0, 2)}
