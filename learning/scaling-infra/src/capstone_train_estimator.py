"""Capstone — Training estimator (LLM training planner).

输入 model + GPU spec → 输出 并行策略 + 显存 + throughput + 时长.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainSpec:
    model_size_b: float
    seq_len: int
    batch: int
    n_token: float
    n_gpu: int
    gpu_vram_gb: float
    gpu_tflops: float
    dtype: str = "bf16"
    hidden: Optional[int] = None
    n_layer: Optional[int] = None


@dataclass
class TrainPlan:
    strategy: str
    mem_per_gpu_gb: float
    feasible: bool
    tok_per_s: float
    hours: float
    cost_usd: Optional[float] = None
    notes: str = ""


BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def _infer_layer_hidden(b: float) -> tuple:
    table = {
        0.5: (24, 1024),
        1.0: (24, 2048),
        3.0: (32, 3072),
        7.0: (32, 4096),
        13.0: (40, 5120),
        30.0: (60, 6656),
        70.0: (80, 8192),
        175.0: (96, 12288),
    }
    closest = min(table.keys(), key=lambda x: abs(x - b))
    return table[closest]


def _activation_gb(batch, seq, hidden, n_layer, bpp=2) -> float:
    return batch * seq * hidden * n_layer * bpp * 10 / 1e9


def estimate(spec: TrainSpec, gpu_hour_usd: float = 1.5) -> TrainPlan:
    n = spec.model_size_b * 1e9
    bpp = BYTES[spec.dtype]
    w = n * bpp / 1e9
    g = w
    opt = n * 4 * 2 / 1e9
    h = spec.hidden or _infer_layer_hidden(spec.model_size_b)[1]
    L = spec.n_layer or _infer_layer_hidden(spec.model_size_b)[0]
    activ = _activation_gb(spec.batch // spec.n_gpu, spec.seq_len, h, L)

    strategies = [
        ("DP", w + g + opt + activ),
        ("ZeRO-1", w + g + opt / spec.n_gpu + activ),
        ("ZeRO-2", w + g / spec.n_gpu + opt / spec.n_gpu + activ),
        ("ZeRO-3/FSDP", (w + g + opt) / spec.n_gpu + activ),
        ("FSDP + grad ckpt",
         (w + g + opt) / spec.n_gpu + activ * 0.2),
        ("TP=8 + FSDP + grad ckpt",
         (w + g + opt) / (spec.n_gpu * 8) + activ * 0.2 / 8),
    ]

    selected = None
    for name, mem in strategies:
        if mem <= spec.gpu_vram_gb * 0.9:
            selected = (name, mem)
            break

    if selected is None:
        return TrainPlan(strategy="INFEASIBLE",
                         mem_per_gpu_gb=strategies[-1][1],
                         feasible=False,
                         tok_per_s=0, hours=0,
                         notes="Need more GPU or PP")

    name, mem = selected
    mfu_target = 0.45
    tok_per_s_per_gpu = mfu_target * spec.gpu_tflops * 1e12 / (6 * n)
    total_tok_s = tok_per_s_per_gpu * spec.n_gpu

    hours = spec.n_token / total_tok_s / 3600
    cost = hours * gpu_hour_usd * spec.n_gpu

    return TrainPlan(
        strategy=name,
        mem_per_gpu_gb=mem,
        feasible=True,
        tok_per_s=total_tok_s,
        hours=hours,
        cost_usd=cost,
        notes=f"Inferred hidden={h} layer={L}",
    )


def report(spec: TrainSpec, plan: TrainPlan) -> str:
    return f"""
Input:
  model={spec.model_size_b}B  seq={spec.seq_len}  batch={spec.batch}
  n_token={spec.n_token:.1e}  n_gpu={spec.n_gpu}×({spec.gpu_vram_gb}GB)
  dtype={spec.dtype}
  {plan.notes}
Plan:
  strategy:       {plan.strategy}
  mem/gpu:        {plan.mem_per_gpu_gb:.1f} GB
  throughput:     {plan.tok_per_s:.0f} tok/s
  time:           {plan.hours:.1f} hours
  est. cost:      ${plan.cost_usd:.0f} (cloud)
"""


if __name__ == "__main__":
    cases = [
        TrainSpec(model_size_b=1.5, seq_len=2048, batch=64,
                   n_token=20e9, n_gpu=1, gpu_vram_gb=24,
                   gpu_tflops=1500),
        TrainSpec(model_size_b=7, seq_len=2048, batch=128,
                   n_token=2e9, n_gpu=8, gpu_vram_gb=80,
                   gpu_tflops=312),
        TrainSpec(model_size_b=70, seq_len=2048, batch=512,
                   n_token=15e12, n_gpu=64, gpu_vram_gb=80,
                   gpu_tflops=312),
        TrainSpec(model_size_b=175, seq_len=2048, batch=2048,
                   n_token=300e9, n_gpu=8, gpu_vram_gb=80,
                   gpu_tflops=312),
    ]
    for s in cases:
        p = estimate(s)
        print(report(s, p))
        print("-" * 60)
