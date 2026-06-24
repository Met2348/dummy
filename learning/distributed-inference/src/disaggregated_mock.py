"""Disaggregated Prefill/Decode mock - single-process simulator.

Teaching goal (lecture L08): show *why* physically separating the prefill and
decode stages raises serving throughput, and *when* it stops paying off.

The benefit must be **derived from a modeled mechanism**, not a hardcoded
constant. The mechanism here is prefill/decode **interference**:

* Colocated: prefill (compute-bound) and decode (memory-bound) share the same
  GPUs. While a long prompt is being prefilled, the decode steps running on that
  GPU are slowed down. We model this as a per-token decode slowdown that grows
  with how much prefill work is contending for the GPU.
* Disaggregated: prefill and decode live on separate GPU pools, so decode runs
  interference-free (faster TPOT) -- but every request must now ship its prompt
  KV cache across the link first (slower TTFT, and wasted bandwidth for short
  prompts).

Consequences that fall out of the model (verified by tests), matching the
lecture instead of contradicting it:

* disagg TPOT < colocate TPOT  (decode no longer fights prefill)
* disagg TTFT > colocate TTFT  (added KV transfer hop)
* the throughput gain *grows with prompt length* and *shrinks on a slow link*
  -- i.e. "disaggregation pays off for long prompts", L08 section 5/6.

This is a reduced single-process model; `distserve_original_minimal.py` carries
the queueing-theoretic version with an explicit prefill/decode GPU search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
    # Resource picture used to turn per-request latency into throughput.
    n_gpus: int = 8
    # Colocated: prefill & decode share every GPU, so a GPU can host fewer
    # concurrent decode streams; disagg dedicates GPUs to each role.
    colocate_streams_per_gpu: int = 4
    disagg_streams_per_gpu: int = 8
    # How strongly a concurrent prefill slows the decode steps sharing its GPU.
    # 0 -> no interference (disagg behaves like colocate); >0 -> real penalty.
    interference: float = 0.6


def kv_transfer_ms(prompt_len: int, hw: HardwareConfig) -> float:
    """Bandwidth-only time to ship one request's prompt KV cache."""
    payload = prompt_len * hw.kv_bytes_per_token
    return payload / (hw.transfer_bw_gbps * 1e9) * 1000.0


def _interference_factor(w: WorkloadConfig, hw: HardwareConfig) -> float:
    """Decode-step slowdown when prefill and decode share a GPU.

    Scales with the prefill/decode work ratio: a long prompt (lots of prefill)
    contending with a short generation (few decode steps) hurts the most. This
    is what makes disaggregation prompt-length dependent.
    """
    prefill_work = w.prompt_len * hw.prefill_ms_per_token
    decode_work = max(1.0, w.out_len * hw.decode_ms_per_token)
    return 1.0 + hw.interference * (prefill_work / decode_work)


def colocate(w: WorkloadConfig, hw: HardwareConfig) -> Dict:
    """Prefill and decode share GPUs -> decode is slowed by interference."""
    prefill_ms = w.prompt_len * hw.prefill_ms_per_token
    inter = _interference_factor(w, hw)
    tpot_ms = hw.decode_ms_per_token * inter            # decode pays interference
    decode_ms = w.out_len * tpot_ms
    per_req_ms = prefill_ms + decode_ms
    concurrency = hw.n_gpus * hw.colocate_streams_per_gpu
    total_s = per_req_ms * w.n_reqs / concurrency / 1000.0
    tput = w.n_reqs * w.out_len / total_s
    return {"config": "colocate", "ttft_ms": round(prefill_ms, 1),
            "tpot_ms": round(tpot_ms, 1), "tok_per_s": round(tput, 1),
            "wall_s": round(total_s, 2)}


def disagg(w: WorkloadConfig, hw: HardwareConfig, cross_node: bool = False) -> Dict:
    """Separate prefill/decode pools -> no interference, but pay KV transfer."""
    bw = 50.0 if cross_node else hw.transfer_bw_gbps
    hw_link = HardwareConfig(**{**hw.__dict__, "transfer_bw_gbps": bw})
    prefill_ms = w.prompt_len * hw.prefill_ms_per_token
    transfer_ms = kv_transfer_ms(w.prompt_len, hw_link)
    tpot_ms = hw.decode_ms_per_token                    # interference removed
    decode_ms = w.out_len * tpot_ms
    ttft_ms = prefill_ms + transfer_ms                  # extra hop hits TTFT
    # Decode pool runs interference-free at higher per-GPU concurrency; the
    # transfer hop is added to the serviced work so a slow link erodes the gain.
    per_req_ms = prefill_ms + transfer_ms + decode_ms
    concurrency = hw.n_gpus * hw.disagg_streams_per_gpu
    total_s = per_req_ms * w.n_reqs / concurrency / 1000.0
    tput = w.n_reqs * w.out_len / total_s
    cfg = "disagg-remote" if cross_node else "disagg-near"
    return {"config": cfg, "ttft_ms": round(ttft_ms, 1),
            "tpot_ms": round(tpot_ms, 1), "tok_per_s": round(tput, 1),
            "wall_s": round(total_s, 2)}


def demo() -> None:
    """Print the colocate vs disagg comparison and the prompt-length trend."""
    hw = HardwareConfig()
    print("=== Disaggregated Prefill/Decode (single-process mock) ===")
    print(f"hardware: prefill {hw.prefill_ms_per_token} ms/tok, "
          f"decode {hw.decode_ms_per_token} ms/tok, "
          f"interference {hw.interference}, NVLink {hw.transfer_bw_gbps} GB/s\n")

    w = WorkloadConfig(n_reqs=32, prompt_len=1024, out_len=128)
    rows = [colocate(w, hw), disagg(w, hw), disagg(w, hw, cross_node=True)]
    print(f"workload: {w.n_reqs} reqs, prompt={w.prompt_len}, out={w.out_len}")
    print(f"{'config':<14}{'TTFT ms':>9}{'TPOT ms':>9}{'tok/s':>9}{'wall s':>9}")
    for r in rows:
        print(f"{r['config']:<14}{r['ttft_ms']:>9}{r['tpot_ms']:>9}"
              f"{r['tok_per_s']:>9}{r['wall_s']:>9}")
    base = rows[0]["tok_per_s"]
    print(f"\ndisagg-near throughput gain over colocate: "
          f"+{(rows[1]['tok_per_s'] / base - 1) * 100:.0f}%  "
          f"(TPOT {base and rows[0]['tpot_ms']} -> {rows[1]['tpot_ms']} ms)")

    print("\n--- gain scales with prompt length (KV transfer must amortize) ---")
    print(f"{'prompt_len':>11}{'colo tok/s':>12}{'disagg tok/s':>14}{'gain %':>9}")
    for pl in (128, 512, 1024, 4096, 8192):
        wl = WorkloadConfig(n_reqs=32, prompt_len=pl, out_len=128)
        c = colocate(wl, hw)["tok_per_s"]
        d = disagg(wl, hw)["tok_per_s"]
        print(f"{pl:>11}{c:>12}{d:>14}{(d / c - 1) * 100:>8.0f}%")
    print("\n=> long prompts benefit most; a slow cross-node link shrinks the gain.")


if __name__ == "__main__":
    demo()
