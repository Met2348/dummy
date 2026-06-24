"""A small DistServe-style goodput simulator.

The real DistServe system uses profiling, a discrete-event simulator, and
placement search over tensor and pipeline parallelism. This file keeps only the
paper's core teaching variables:

* TTFT: time to first token, dominated by prefill.
* TPOT: time per output token, dominated by decode.
* KV transfer: the extra cost introduced by disaggregation.
* SLO attainment: fraction of requests meeting both latency limits.
* per-GPU goodput: request rate that satisfies the SLO target divided by GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Request:
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SLO:
    ttft_ms: float
    tpot_ms: float


@dataclass(frozen=True)
class Profile:
    prefill_ms_per_token: float = 0.26
    decode_ms_per_token: float = 8.0
    kv_bytes_per_token: int = 2_200_000
    decode_prefill_interference: float = 0.34


@dataclass(frozen=True)
class Placement:
    name: str
    colocated: bool
    prefill_gpus: int
    decode_gpus: int
    bandwidth_gbps: float = 900.0

    @property
    def total_gpus(self) -> int:
        if self.colocated:
            return max(self.prefill_gpus, self.decode_gpus)
        return self.prefill_gpus + self.decode_gpus


@dataclass(frozen=True)
class Latency:
    ttft_ms: float
    tpot_ms: float


@dataclass(frozen=True)
class GoodputResult:
    placement: Placement
    request_rate_rps: float
    attainment: float
    goodput_rps: float
    per_gpu_goodput_rps: float


def kv_transfer_ms(prompt_tokens: int, profile: Profile, bandwidth_gbps: float) -> float:
    """Bandwidth-only transfer time for the prompt KV cache."""
    payload_bytes = prompt_tokens * profile.kv_bytes_per_token
    return payload_bytes * 8.0 / (bandwidth_gbps * 1e9) * 1000.0


def _parallel_speedup(gpus: int, exponent: float) -> float:
    """A small diminishing-return model for model parallelism."""
    return max(1.0, float(gpus) ** exponent)


def _md1_wait_ms(service_ms: float, arrival_rate_rps: float) -> float:
    """Mean waiting time for an M/D/1 queue; infinite when overloaded."""
    rho = arrival_rate_rps * service_ms / 1000.0
    if rho >= 0.98:
        return float("inf")
    return (arrival_rate_rps * service_ms * service_ms / 1000.0) / (2.0 * (1.0 - rho))


def _mean_request(requests: Sequence[Request]) -> Request:
    return Request(
        prompt_tokens=max(1, round(mean(r.prompt_tokens for r in requests))),
        output_tokens=max(1, round(mean(r.output_tokens for r in requests))),
    )


def estimate_latencies(
    requests: Sequence[Request],
    placement: Placement,
    request_rate_rps: float,
    profile: Profile | None = None,
) -> List[Latency]:
    """Estimate TTFT and TPOT for every request under a placement."""
    if not requests:
        return []

    p = profile or Profile()
    avg = _mean_request(requests)

    if placement.colocated:
        gpus = placement.total_gpus
        prefill_speed = _parallel_speedup(gpus, 0.78)
        decode_speed = _parallel_speedup(gpus, 0.35)
        avg_prefill_ms = avg.prompt_tokens * p.prefill_ms_per_token / prefill_speed
        avg_decode_total_ms = avg.output_tokens * p.decode_ms_per_token / decode_speed
        shared_wait_ms = _md1_wait_ms(avg_prefill_ms + avg_decode_total_ms, request_rate_rps)

        result: List[Latency] = []
        for req in requests:
            prefill_ms = req.prompt_tokens * p.prefill_ms_per_token / prefill_speed
            decode_step_ms = p.decode_ms_per_token / decode_speed
            interference = 1.0 + p.decode_prefill_interference * (req.prompt_tokens / 1024.0)
            ttft_ms = shared_wait_ms + prefill_ms * 1.08
            tpot_ms = decode_step_ms * interference + shared_wait_ms / max(1, req.output_tokens)
            result.append(Latency(ttft_ms=ttft_ms, tpot_ms=tpot_ms))
        return result

    prefill_speed = _parallel_speedup(placement.prefill_gpus, 0.82)
    decode_speed = _parallel_speedup(placement.decode_gpus, 0.45)
    avg_prefill_ms = avg.prompt_tokens * p.prefill_ms_per_token / prefill_speed
    avg_decode_total_ms = avg.output_tokens * p.decode_ms_per_token / decode_speed
    prefill_wait_ms = _md1_wait_ms(avg_prefill_ms, request_rate_rps)
    decode_wait_ms = _md1_wait_ms(avg_decode_total_ms, request_rate_rps)

    result = []
    for req in requests:
        prefill_ms = req.prompt_tokens * p.prefill_ms_per_token / prefill_speed
        transfer_ms = kv_transfer_ms(req.prompt_tokens, p, placement.bandwidth_gbps)
        decode_step_ms = p.decode_ms_per_token / decode_speed
        ttft_ms = prefill_wait_ms + prefill_ms + transfer_ms
        tpot_ms = decode_step_ms + decode_wait_ms / max(1, req.output_tokens)
        result.append(Latency(ttft_ms=ttft_ms, tpot_ms=tpot_ms))
    return result


def slo_attainment(latencies: Iterable[Latency], slo: SLO) -> float:
    items = list(latencies)
    if not items:
        return 0.0
    passed = sum(1 for x in items if x.ttft_ms <= slo.ttft_ms and x.tpot_ms <= slo.tpot_ms)
    return passed / len(items)


def goodput_at_rate(
    requests: Sequence[Request],
    placement: Placement,
    slo: SLO,
    request_rate_rps: float,
    profile: Profile | None = None,
) -> GoodputResult:
    attainment = slo_attainment(estimate_latencies(requests, placement, request_rate_rps, profile), slo)
    goodput = request_rate_rps * attainment
    return GoodputResult(
        placement=placement,
        request_rate_rps=request_rate_rps,
        attainment=attainment,
        goodput_rps=goodput,
        per_gpu_goodput_rps=goodput / placement.total_gpus,
    )


def max_goodput(
    requests: Sequence[Request],
    placement: Placement,
    slo: SLO,
    candidate_rates: Sequence[float],
    attainment_target: float = 0.9,
    profile: Profile | None = None,
) -> GoodputResult:
    """Choose the highest request rate that satisfies the target attainment."""
    best = goodput_at_rate(requests, placement, slo, 0.0, profile)
    for rate in candidate_rates:
        current = goodput_at_rate(requests, placement, slo, rate, profile)
        if current.attainment >= attainment_target and current.goodput_rps >= best.goodput_rps:
            best = current
    return best


def search_gpu_split(
    total_gpus: int,
    requests: Sequence[Request],
    slo: SLO,
    candidate_rates: Sequence[float],
    bandwidth_gbps: float = 900.0,
    attainment_target: float = 0.9,
    profile: Profile | None = None,
) -> GoodputResult:
    """Search a DistServe-like prefill/decode GPU split."""
    if total_gpus < 2:
        raise ValueError("disaggregated placement needs at least two GPUs")

    best: GoodputResult | None = None
    for prefill_gpus in range(1, total_gpus):
        decode_gpus = total_gpus - prefill_gpus
        placement = Placement(
            name=f"distserve-p{prefill_gpus}-d{decode_gpus}",
            colocated=False,
            prefill_gpus=prefill_gpus,
            decode_gpus=decode_gpus,
            bandwidth_gbps=bandwidth_gbps,
        )
        result = max_goodput(requests, placement, slo, candidate_rates, attainment_target, profile)
        if best is None or result.per_gpu_goodput_rps > best.per_gpu_goodput_rps:
            best = result
    assert best is not None
    return best


def colocated_placement(total_gpus: int) -> Placement:
    return Placement(
        name=f"colocated-{total_gpus}gpu",
        colocated=True,
        prefill_gpus=total_gpus,
        decode_gpus=total_gpus,
    )


def demo() -> None:
    """DistServe-style goodput: colocated vs disaggregated + a P/D GPU split search."""
    requests = [
        Request(prompt_tokens=1024, output_tokens=96),
        Request(prompt_tokens=1536, output_tokens=96),
        Request(prompt_tokens=2048, output_tokens=128),
        Request(prompt_tokens=1792, output_tokens=96),
    ]
    slo = SLO(ttft_ms=900, tpot_ms=13)
    rates = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    total_gpus = 3

    print("=== DistServe-style goodput simulator ===")
    print(f"workload: {len(requests)} request types, prompts "
          f"{[r.prompt_tokens for r in requests]}, total_gpus={total_gpus}")
    print(f"SLO: TTFT <= {slo.ttft_ms} ms, TPOT <= {slo.tpot_ms} ms\n")

    colo = colocated_placement(total_gpus)
    colo_best = max_goodput(requests, colo, slo, rates)
    disagg_best = search_gpu_split(total_gpus, requests, slo, rates)

    print(f"{'placement':>22}{'rate rps':>10}{'attain':>9}{'goodput':>9}{'per-GPU':>9}")
    for tag, res in (("colocated", colo_best), (disagg_best.placement.name, disagg_best)):
        print(f"{tag:>22}{res.request_rate_rps:>10.2f}{res.attainment:>9.0%}"
              f"{res.goodput_rps:>9.2f}{res.per_gpu_goodput_rps:>9.3f}")

    gain = (disagg_best.per_gpu_goodput_rps / max(colo_best.per_gpu_goodput_rps, 1e-9) - 1) * 100
    print(f"\nbest split = {disagg_best.placement.prefill_gpus} prefill + "
          f"{disagg_best.placement.decode_gpus} decode GPU(s); "
          f"per-GPU goodput +{gain:.0f}% vs colocated.")

    print("\n--- a slow cross-node link inflates TTFT ---")
    one = [Request(prompt_tokens=2048, output_tokens=64)]
    for bw in (900.0, 50.0, 25.0):
        pl = Placement(f"p2-d1@{bw:g}", colocated=False, prefill_gpus=2,
                       decode_gpus=1, bandwidth_gbps=bw)
        ttft = estimate_latencies(one, pl, request_rate_rps=0.5)[0].ttft_ms
        print(f"  link {bw:>5g} GB/s: TTFT = {ttft:6.1f} ms  "
              f"({'OK' if ttft <= slo.ttft_ms else 'SLO VIOLATED'})")


if __name__ == "__main__":
    demo()
