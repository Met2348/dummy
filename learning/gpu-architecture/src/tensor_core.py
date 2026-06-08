"""Tensor Core MMA shape model: H100 wgmma / B200 tcgen05."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MMAShape:
    name: str
    m: int
    n: int
    k: int
    dtype: str
    cycles: int       # per warp-group on H100, conceptual on B200


H100_WGMMA = [
    MMAShape("wgmma.m64n8k16.bf16",   64,   8, 16, "bf16",   32),
    MMAShape("wgmma.m64n256k16.bf16", 64, 256, 16, "bf16",   32),
    MMAShape("wgmma.m64n256k32.fp8",  64, 256, 32, "fp8",    32),
]

B200_TCGEN05 = [
    MMAShape("tcgen05.m128n256k32.bf16", 128, 256, 32, "bf16", 32),
    MMAShape("tcgen05.m128n256k64.fp8",  128, 256, 64, "fp8",  32),
    MMAShape("tcgen05.m128n256k128.fp4", 128, 256, 128, "fp4", 32),
]


def flops_per_cycle(shape: MMAShape) -> int:
    return 2 * shape.m * shape.n * shape.k


def pick_shape(target_m: int, target_n: int, dtype: str, isa: list[MMAShape]) -> MMAShape:
    """Pick the MMA shape that best tiles target_m by target_n."""
    valid = [s for s in isa if s.dtype == dtype]
    if not valid:
        raise ValueError(f"no shape for dtype {dtype}")

    def score(s: MMAShape) -> tuple[int, int]:
        m_waste = (target_m % s.m) if target_m >= s.m else (s.m - target_m)
        n_waste = (target_n % s.n) if target_n >= s.n else (s.n - target_n)
        return (m_waste + n_waste, -flops_per_cycle(s))

    return sorted(valid, key=score)[0]


def _self_test() -> None:
    s = pick_shape(64, 256, "bf16", H100_WGMMA)
    assert s.name == "wgmma.m64n256k16.bf16"
    s8 = pick_shape(64, 256, "fp8", H100_WGMMA)
    assert s8.dtype == "fp8"
    # FP4 only on B200
    s4 = pick_shape(128, 256, "fp4", B200_TCGEN05)
    assert s4.dtype == "fp4"
    # FP4 has 4x throughput of FP8 at same shape
    fp8_b200 = pick_shape(128, 256, "fp8", B200_TCGEN05)
    assert flops_per_cycle(s4) >= 2 * flops_per_cycle(fp8_b200)
    print(f"[OK] tensor_core (FP4 {flops_per_cycle(s4)} FLOP/cycle)")


if __name__ == "__main__":
    _self_test()
