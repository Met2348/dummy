"""Roofline analyzer for common LLM ops."""
from __future__ import annotations
from dataclasses import dataclass
from common import GPUS, GPUSpec, roofline_flops


@dataclass
class OpProfile:
    name: str
    flops: int           # total operations
    bytes_moved: int     # HBM traffic

    def ai(self) -> float:
        return self.flops / max(1, self.bytes_moved)


def gemm_profile(m: int, n: int, k: int, dtype_bytes: int = 2) -> OpProfile:
    flops = 2 * m * n * k
    bytes_moved = dtype_bytes * (m * k + k * n + m * n)
    return OpProfile(f"GEMM({m}x{n}x{k})", flops, bytes_moved)


def attention_profile(b: int, h: int, s: int, d: int, dtype_bytes: int = 2) -> OpProfile:
    qkv = 3 * b * h * s * d
    flops = 4 * b * h * s * s * d                # QK^T + softmax*V
    bytes_moved = dtype_bytes * (qkv + b * h * s * s + b * h * s * d)
    return OpProfile(f"Attn(b{b},h{h},s{s},d{d})", flops, bytes_moved)


def layernorm_profile(n_tokens: int, hidden: int, dtype_bytes: int = 2) -> OpProfile:
    flops = 8 * n_tokens * hidden
    bytes_moved = 2 * dtype_bytes * n_tokens * hidden
    return OpProfile(f"LayerNorm(n{n_tokens},h{hidden})", flops, bytes_moved)


def analyze(op: OpProfile, gpu: GPUSpec) -> dict:
    ai = op.ai()
    achievable = roofline_flops(gpu, ai)
    bound = "compute" if ai >= gpu.ridge_point_bf16() else "memory"
    return {
        "op": op.name,
        "ai": round(ai, 2),
        "achievable_tflops": round(achievable, 1),
        "peak_tflops": gpu.bf16_tflops,
        "utilization_pct": round(100 * achievable / gpu.bf16_tflops, 1),
        "bound_by": bound,
    }


def _self_test() -> None:
    h100 = GPUS["H100"]
    gemv = gemm_profile(1, 4096, 4096)
    big_gemm = gemm_profile(4096, 4096, 4096)
    ln = layernorm_profile(4096, 4096)
    attn = attention_profile(1, 32, 2048, 128)

    r_gemv = analyze(gemv, h100)
    r_big = analyze(big_gemm, h100)
    r_ln = analyze(ln, h100)
    r_attn = analyze(attn, h100)

    assert r_gemv["bound_by"] == "memory"
    assert r_big["bound_by"] == "compute"
    assert r_ln["bound_by"] == "memory"
    assert r_big["utilization_pct"] > 50.0, r_big
    assert r_ln["utilization_pct"] < 5.0, r_ln       # LayerNorm is awful
    print(f"[OK] roofline analyzer (big GEMM util {r_big['utilization_pct']}%)")


if __name__ == "__main__":
    _self_test()
