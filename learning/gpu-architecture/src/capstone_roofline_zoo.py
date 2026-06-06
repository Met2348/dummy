"""Capstone: roofline zoo — 10 ops × 4 GPUs, identify bottlenecks."""
from __future__ import annotations
from common import GPUS
from roofline import gemm_profile, attention_profile, layernorm_profile, analyze


WORKLOADS = [
    ("gemv-1x4k-4k", gemm_profile(1, 4096, 4096)),
    ("gemm-2k-2k-2k", gemm_profile(2048, 2048, 2048)),
    ("gemm-8k-8k-8k", gemm_profile(8192, 8192, 8192)),
    ("attn-b1-h32-s2k", attention_profile(1, 32, 2048, 128)),
    ("attn-b1-h32-s32k", attention_profile(1, 32, 32768, 128)),
    ("layernorm-4k-4k", layernorm_profile(4096, 4096)),
    ("layernorm-4k-8k", layernorm_profile(4096, 8192)),
    ("gemm-4k-4k-128", gemm_profile(4096, 4096, 128)),
    ("gemm-128-4k-4k", gemm_profile(128, 4096, 4096)),
    ("gemm-32k-32k-32k", gemm_profile(32768, 32768, 32768)),
]


def run() -> list[dict]:
    results = []
    for gpu_name in ["A100", "H100", "H200", "B200"]:
        gpu = GPUS[gpu_name]
        for op_name, op in WORKLOADS:
            r = analyze(op, gpu)
            r["gpu"] = gpu_name
            r["op"] = op_name
            results.append(r)
    return results


def summarize(results: list[dict]) -> dict:
    n_memory = sum(1 for r in results if r["bound_by"] == "memory")
    n_compute = len(results) - n_memory
    avg_util = sum(r["utilization_pct"] for r in results) / len(results)
    h100_big_gemm = [r for r in results
                     if r["gpu"] == "H100" and r["op"] == "gemm-8k-8k-8k"][0]
    return {
        "total": len(results),
        "memory_bound": n_memory,
        "compute_bound": n_compute,
        "avg_util_pct": round(avg_util, 1),
        "h100_big_gemm_util": h100_big_gemm["utilization_pct"],
    }


def _self_test() -> None:
    results = run()
    s = summarize(results)
    assert s["total"] == 40
    assert s["memory_bound"] > s["compute_bound"], s   # most LLM ops are mem-bound
    assert s["h100_big_gemm_util"] > 80.0, s
    print(f"[OK] capstone_roofline_zoo: {s['memory_bound']}/{s['total']} mem-bound, "
          f"H100 big GEMM {s['h100_big_gemm_util']}% util")


if __name__ == "__main__":
    _self_test()
