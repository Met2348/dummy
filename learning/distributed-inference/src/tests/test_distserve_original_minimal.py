import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distserve_original_minimal import (
    Placement,
    Profile,
    Request,
    SLO,
    colocated_placement,
    estimate_latencies,
    goodput_at_rate,
    kv_transfer_ms,
    search_gpu_split,
)


def _long_prompt_workload():
    return [
        Request(prompt_tokens=1024, output_tokens=96),
        Request(prompt_tokens=1536, output_tokens=96),
        Request(prompt_tokens=2048, output_tokens=128),
        Request(prompt_tokens=1792, output_tokens=96),
    ]


def test_kv_transfer_scales_with_prompt_and_bandwidth():
    p = Profile()
    base = kv_transfer_ms(512, p, bandwidth_gbps=900)
    longer = kv_transfer_ms(1024, p, bandwidth_gbps=900)
    slower = kv_transfer_ms(512, p, bandwidth_gbps=50)

    assert longer == base * 2
    assert slower > base * 10


def test_disaggregation_improves_tpot_under_prefill_decode_interference():
    requests = _long_prompt_workload()
    slo = SLO(ttft_ms=900, tpot_ms=13)
    rate = 1.1
    colocated = colocated_placement(total_gpus=3)
    distserve = Placement("distserve-p1-d2", colocated=False, prefill_gpus=1, decode_gpus=2)

    colocated_result = goodput_at_rate(requests, colocated, slo, rate)
    distserve_result = goodput_at_rate(requests, distserve, slo, rate)

    assert distserve_result.attainment >= 0.9
    assert colocated_result.attainment < distserve_result.attainment


def test_search_prefers_more_prefill_gpus_when_ttft_is_tight():
    requests = [
        Request(prompt_tokens=4096, output_tokens=32),
        Request(prompt_tokens=3072, output_tokens=32),
        Request(prompt_tokens=4096, output_tokens=48),
        Request(prompt_tokens=3584, output_tokens=32),
    ]
    slo = SLO(ttft_ms=950, tpot_ms=14)
    rates = [0.2, 0.3, 0.4, 0.5, 0.6]

    best = search_gpu_split(total_gpus=3, requests=requests, slo=slo, candidate_rates=rates)

    assert best.placement.prefill_gpus == 2
    assert best.placement.decode_gpus == 1
    assert best.attainment >= 0.9


def test_slow_cross_node_link_can_break_ttft_slo():
    requests = [Request(prompt_tokens=2048, output_tokens=64)]
    slo = SLO(ttft_ms=700, tpot_ms=20)
    near = Placement("near", colocated=False, prefill_gpus=2, decode_gpus=1, bandwidth_gbps=900)
    far = Placement("far", colocated=False, prefill_gpus=2, decode_gpus=1, bandwidth_gbps=25)

    near_latency = estimate_latencies(requests, near, request_rate_rps=0.5)[0]
    far_latency = estimate_latencies(requests, far, request_rate_rps=0.5)[0]

    assert near_latency.ttft_ms <= slo.ttft_ms
    assert far_latency.ttft_ms > near_latency.ttft_ms
