"""Original Roofline paper concepts in a small runnable model.

The 2009 paper uses "operational intensity": useful operations per DRAM byte
after the cache hierarchy has filtered traffic. This module keeps that term
and adds the paper's second important idea: ceilings that show which
optimization must be performed before a kernel can reach a higher roof.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineRoofline:
    name: str
    peak_gflops: float
    memory_gb_s: float

    def ridge_point(self) -> float:
        """Operations per DRAM byte needed to reach peak compute."""
        return self.peak_gflops / self.memory_gb_s

    def attainable_gflops(self, operational_intensity: float) -> float:
        """Roofline bound: min(peak compute, memory BW times intensity)."""
        return min(self.peak_gflops, self.memory_gb_s * operational_intensity)

    def bound_type(self, operational_intensity: float) -> str:
        return "compute" if operational_intensity >= self.ridge_point() else "memory"


@dataclass(frozen=True)
class KernelObservation:
    name: str
    operations: float
    dram_bytes: float

    def operational_intensity(self) -> float:
        if self.dram_bytes <= 0:
            raise ValueError("dram_bytes must be positive")
        return self.operations / self.dram_bytes


@dataclass(frozen=True)
class Ceiling:
    name: str
    kind: str
    value: float
    next_action: str


OPTERON_X2 = MachineRoofline("AMD Opteron X2 2214", 17.6, 15.0)
OPTERON_X4 = MachineRoofline("AMD Opteron X4 Barcelona", 74.0, 16.6)


PAPER_KERNELS = {
    "spmv_after_blocking": KernelObservation("SpMV after register blocking", 25.0, 100.0),
    "lbmhd_no_allocate": KernelObservation("LBMHD with no-allocate store", 107.0, 100.0),
    "stencil_write_allocate": KernelObservation("3-D stencil, write allocate", 50.0, 100.0),
    "fft_128_cube": KernelObservation("3-D FFT, 128 cube", 164.0, 100.0),
}


def attainable_with_ceilings(
    machine: MachineRoofline,
    kernel: KernelObservation,
    ceilings: list[Ceiling],
) -> dict:
    """Apply compute and memory ceilings to the basic roofline bound."""
    compute_roof = machine.peak_gflops
    memory_roof = machine.memory_gb_s
    actions = []

    for ceiling in ceilings:
        if ceiling.kind == "compute":
            compute_roof = min(compute_roof, ceiling.value)
        elif ceiling.kind == "memory":
            memory_roof = min(memory_roof, ceiling.value)
        else:
            raise ValueError(f"unknown ceiling kind: {ceiling.kind}")
        actions.append(ceiling.next_action)

    oi = kernel.operational_intensity()
    attainable = min(compute_roof, memory_roof * oi)
    bottleneck = "compute" if compute_roof <= memory_roof * oi else "memory"
    return {
        "kernel": kernel.name,
        "machine": machine.name,
        "operational_intensity": round(oi, 3),
        "attainable_gflops": round(attainable, 3),
        "bottleneck": bottleneck,
        "blocked_by": [c.name for c in ceilings],
        "next_actions": actions,
    }


def recommend_optimizations(machine: MachineRoofline, kernel: KernelObservation) -> list[str]:
    """Return the paper-style optimization order for a kernel position."""
    oi = kernel.operational_intensity()
    recs: list[str] = []

    if oi < machine.ridge_point():
        recs.extend([
            "first reduce DRAM traffic: blocking, padding, no-allocate stores, compression",
            "then improve bandwidth: unit-stride access, memory affinity, prefetching",
        ])
    else:
        recs.extend([
            "first improve in-core execution: ILP, unrolling, SIMD or tensor cores",
            "then check operation mix and issue balance",
        ])

    if oi < 1.0:
        recs.append("treat peak FLOPS as mostly irrelevant until intensity rises")
    else:
        recs.append("check both memory and compute ceilings because the kernel is near the crossover")

    return recs


def improve_operational_intensity(
    kernel: KernelObservation,
    *,
    traffic_reduction_factor: float,
) -> KernelObservation:
    """Model cache or data-structure work that reduces DRAM bytes."""
    if traffic_reduction_factor <= 0:
        raise ValueError("traffic_reduction_factor must be positive")
    return KernelObservation(
        name=f"{kernel.name} after traffic reduction",
        operations=kernel.operations,
        dram_bytes=kernel.dram_bytes / traffic_reduction_factor,
    )


def _self_test() -> None:
    assert 1.1 < OPTERON_X2.ridge_point() < 1.3
    assert 4.3 < OPTERON_X4.ridge_point() < 4.6

    spmv = PAPER_KERNELS["spmv_after_blocking"]
    fft = PAPER_KERNELS["fft_128_cube"]
    assert round(spmv.operational_intensity(), 2) == 0.25
    assert OPTERON_X4.bound_type(spmv.operational_intensity()) == "memory"
    assert OPTERON_X4.bound_type(10.0) == "compute"
    assert OPTERON_X4.attainable_gflops(spmv.operational_intensity()) < 5.0
    assert OPTERON_X4.attainable_gflops(10.0) == OPTERON_X4.peak_gflops

    no_affinity = Ceiling(
        name="no memory affinity",
        kind="memory",
        value=7.0,
        next_action="place data near the socket or GPU that consumes it",
    )
    no_simd = Ceiling(
        name="no SIMD",
        kind="compute",
        value=37.0,
        next_action="vectorize or use tensor-core-style matrix instructions",
    )
    bounded = attainable_with_ceilings(OPTERON_X4, fft, [no_affinity, no_simd])
    assert bounded["bottleneck"] == "memory", bounded
    assert bounded["attainable_gflops"] < OPTERON_X4.attainable_gflops(
        fft.operational_intensity()
    )

    improved = improve_operational_intensity(spmv, traffic_reduction_factor=2.0)
    assert improved.operational_intensity() == 2 * spmv.operational_intensity()
    assert "DRAM traffic" in recommend_optimizations(OPTERON_X4, spmv)[0]
    print(f"[OK] roofline_original_minimal (X4 ridge {OPTERON_X4.ridge_point():.2f})")


if __name__ == "__main__":
    _self_test()
