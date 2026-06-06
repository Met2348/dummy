"""Common GPU model — H100 / H200 / B200 / GB200 specs."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    """Vendor-published peak numbers (calibrated 2026-06)."""
    name: str
    bf16_tflops: float          # dense BF16
    fp8_tflops: float           # E4M3
    fp4_tflops: float           # Blackwell only
    hbm_gb: int
    hbm_tb_s: float             # memory bandwidth
    nvlink_tb_s: float          # per-GPU bidirectional
    tdp_w: int

    def ridge_point_bf16(self) -> float:
        """FLOP/byte at which compute and memory roofs meet (BF16)."""
        return (self.bf16_tflops * 1e12) / (self.hbm_tb_s * 1e12)


GPUS: dict[str, GPUSpec] = {
    "A100":   GPUSpec("A100-80G",   312.0,    0.0,    0.0,  80, 2.039, 0.6, 400),
    "H100":   GPUSpec("H100-SXM5",  989.0,  1979.0,    0.0,  80, 3.35,  0.9, 700),
    "H200":   GPUSpec("H200-SXM",   989.0,  1979.0,    0.0, 141, 4.8,   0.9, 700),
    "B200":   GPUSpec("B200",      2250.0,  4500.0, 9000.0, 192, 8.0,   1.8, 1000),
    "GB200":  GPUSpec("GB200-NVL", 2500.0,  5000.0,10000.0, 192, 8.0,   1.8, 1200),
}


def roofline_flops(spec: GPUSpec, ai: float, dtype: str = "bf16") -> float:
    """Achievable TFLOPS at arithmetic intensity `ai` (FLOP/byte)."""
    peak = {"bf16": spec.bf16_tflops, "fp8": spec.fp8_tflops, "fp4": spec.fp4_tflops}[dtype]
    mem_bound = spec.hbm_tb_s * ai            # TB/s * FLOP/byte = TFLOP/s
    return min(peak, mem_bound)


def _self_test() -> None:
    h100 = GPUS["H100"]
    rp = h100.ridge_point_bf16()
    assert 280 < rp < 310, rp                          # ~295 FLOP/byte
    # GEMV (ai ~ 1): memory bound
    assert roofline_flops(h100, 1.0) < 5.0
    # Big GEMM (ai = 500): compute bound
    assert roofline_flops(h100, 500.0) == 989.0
    print(f"[OK] gpu_architecture.common (H100 ridge {rp:.0f} FLOP/byte)")


if __name__ == "__main__":
    _self_test()
