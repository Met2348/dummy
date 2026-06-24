"""Capstone — quantization zoo that runs every method on one toy layer.

This used to be a hard-coded table of published Llama-7B numbers. That looked
tidy but demonstrated nothing: none of the int8 / GPTQ / AWQ / NF4 / FP8 /
SmoothQuant modules in this directory were ever invoked, and the "compression"
and "memory" columns were literals typed by hand.

Now the zoo builds a single toy weight layer ``y = W x`` with correlated
calibration activations and pushes ``W`` through each real quantizer:

- ``error``        — output-reconstruction MSE ``mean((Wq x - W x)^2)``: the
  layer-output error the GPTQ / AWQ papers actually minimise. Computed, not
  quoted.
- ``bits``         — the real weight bit-width each method stores.
- ``compression``  — ``16 / bits`` derived from that bit-width.
- ``mem_mib``      — ``base_params * bits / 8`` for a stated parameter budget,
  so memory follows from the bit-width instead of being a magic number.

A few channels in the calibration set carry much larger activation magnitude
("salient channels") so AWQ's activation-aware scaling and SmoothQuant's
weight<->activation migration have something real to do — otherwise they would
collapse to plain round-to-nearest and the table would mislead.

Run:
    .\\.venv\\Scripts\\python.exe learning\\quantization-deploy\\src\\capstone_quant_zoo.py
"""
from __future__ import annotations

from typing import List, Dict
import json

import torch

from int8_basics import (
    quantize_per_tensor,
    dequantize_per_tensor,
    quantize_per_channel,
    dequantize_per_channel,
)
from gptq_minimal import gptq_quantize, calibrate_hessian
from awq_minimal import search_scales
from smooth_quant import find_smooth_scale, apply_smoothing
from fp8_demo import fp8_round
from bnb_int4 import quantize_nf4, dequantize_nf4


# Stated parameter budget for the "memory" column. The toy layer is tiny; we
# report what these same bit-widths would cost for a fixed budget so the column
# is a real function of `bits` rather than a hand-tuned literal.
BASE_PARAMS = 7_000_000_000  # 7B, matching the Llama-7B framing in the lectures


def build_layer(seed: int = 0):
    """One toy linear layer with correlated, partly-salient activations.

    Returns:
        W: [out, in] weight matrix
        X: [N, in] calibration activations (a few channels are salient)
    """
    torch.manual_seed(seed)
    W = torch.randn(64, 128)
    Z = torch.randn(512, 128)
    mixing = torch.eye(128) + 0.15 * torch.randn(128, 128)
    X = Z @ mixing
    # Salient channels: AWQ / SmoothQuant exist precisely to handle activation
    # outliers. Without them both degrade to round-to-nearest.
    for channel in (7, 33, 91):
        X[:, channel] *= 12.0
    return W, X


def _output_mse(W_q: torch.Tensor, W: torch.Tensor, X: torch.Tensor) -> float:
    ref = W.float() @ X.float().t()
    approx = W_q.float() @ X.float().t()
    return float(((approx - ref) ** 2).mean().item())


# ---- one quantizer per variant; each returns a dequantized weight ----------

def _q_int8_pc(W, X):
    q, s = quantize_per_channel(W, axis=0, n_bits=8)
    return dequantize_per_channel(q, s)


def _q_gptq_4bit(W, X):
    H = calibrate_hessian(X)
    W_q, _ = gptq_quantize(W, H, n_bits=4)
    return W_q


def _q_awq_4bit(W, X):
    s = search_scales(W, X, n_bits=4)          # activation-aware per-channel
    q, scale = quantize_per_channel(W * s, axis=0, n_bits=4)
    return dequantize_per_channel(q, scale) / s


def _q_nf4(W, X):
    idx, scales = quantize_nf4(W, block_size=64)
    return dequantize_nf4(idx, scales, block_size=64, orig_shape=W.shape)


def _q_fp8_e4m3(W, X):
    return fp8_round(W)


def _q_smoothquant_int8(W, X):
    """SmoothQuant: migrate activation outliers into weights, then int8 A*W.

    smooth_quant's convention is ``Y = X[N,K] @ W2[K,M]``; our layer is
    ``y = W[out,in] @ x`` so we smooth ``W2 = W.T`` against the activations and
    quantize both operands. We fold the result back to a [out, in] effective
    weight (W2_sq applied to dequantized smoothed activations) by returning the
    smoothed-and-quantized weight transposed; output MSE is measured the same
    way as every other variant for a fair comparison.
    """
    W2 = W.t().contiguous()                       # [in, out]
    s = find_smooth_scale(X, W2, alpha=0.5)       # [in]
    _, W2_s = apply_smoothing(X, W2, s)
    q, scale = quantize_per_channel(W2_s, axis=1, n_bits=8)
    W2_sq = dequantize_per_channel(q, scale)
    # undo the per-channel smoothing on the weight to recover an effective W
    W_eff = (W2_sq / s.unsqueeze(-1)).t().contiguous()
    return W_eff


# variant registry: (name, bits, quantizer or None for fp16 reference)
_SPECS = [
    ("fp16",             16, None),
    ("int8 (pc)",         8, _q_int8_pc),
    ("GPTQ-4bit",         4, _q_gptq_4bit),
    ("AWQ-4bit",          4, _q_awq_4bit),
    ("NF4 (bnb)",         4, _q_nf4),
    ("FP8 (E4M3)",        8, _q_fp8_e4m3),
    ("SmoothQuant-int8",  8, _q_smoothquant_int8),
]


def run_all(seed: int = 0) -> List[Dict]:
    """Quantize one toy layer every way and return per-variant real metrics."""
    W, X = build_layer(seed=seed)
    rows: List[Dict] = []
    for name, bits, quantizer in _SPECS:
        if quantizer is None:
            err = 0.0
        else:
            err = _output_mse(quantizer(W, X), W, X)
        rows.append(
            dict(
                variant=name,
                bits=bits,
                error=round(err, 4),
                compression=round(16.0 / bits, 2),
                mem_mib=round(BASE_PARAMS * bits / 8 / (1024 ** 2), 1),
            )
        )
    return rows


def best_for(metric: str, rows: List[Dict] | None = None) -> Dict:
    rows = rows if rows is not None else run_all()
    if metric == "accuracy":   # lowest reconstruction error wins
        return min(rows, key=lambda r: r["error"])
    if metric == "memory":     # fewest bytes wins
        return min(rows, key=lambda r: r["mem_mib"])
    if metric == "compression":
        return max(rows, key=lambda r: r["compression"])
    raise ValueError(metric)


def best_quantized_for(metric: str, rows: List[Dict] | None = None) -> Dict:
    """Like best_for but excludes the fp16 reference (error==0 by construction)."""
    rows = rows if rows is not None else run_all()
    quantized = [r for r in rows if r["variant"] != "fp16"]
    if metric == "accuracy":
        return min(quantized, key=lambda r: r["error"])
    return best_for(metric, quantized)


def _zoo_table(rows: List[Dict]) -> str:
    head = "| variant | bits | recon-MSE | compression | mem(MiB) |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['variant']} | {r['bits']} | {r['error']} | "
        f"{r['compression']}x | {r['mem_mib']} |"
        for r in rows
    )
    return head + "\n" + body


if __name__ == "__main__":
    rows = run_all()
    print(_zoo_table(rows))
    print()
    print("lowest recon-MSE (excl fp16):", best_quantized_for("accuracy")["variant"])
    print("best compression:           ", best_for("compression")["variant"])
    print("smallest memory:            ", best_for("memory")["variant"])
    print("\nJSON:")
    print(json.dumps(rows, indent=2))
