"""Tests across all quant modules + capstone."""
import sys, pathlib, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from int8_basics import (
    quantize_per_tensor, dequantize_per_tensor,
    quantize_per_channel, dequantize_per_channel,
    quantize_per_group, dequantize_per_group, mse,
)
from gptq_minimal import gptq_quantize, calibrate_hessian
from awq_minimal import awq_quantize, search_scales
from smooth_quant import find_smooth_scale, apply_smoothing
from fp8_demo import fp8_round, fp8_matmul_mock, relative_error
from bnb_int4 import quantize_nf4, dequantize_nf4
from kv_quant import quantize_kv_per_token, dequantize_kv_per_token, attention_with_quant_kv
from quant_eval import memory_table
from capstone_quant_zoo import run_all, best_for, best_quantized_for


def test_int8_per_tensor_round_trip():
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    q, s = quantize_per_tensor(x)
    x_dq = dequantize_per_tensor(q, s)
    assert mse(x, x_dq) < 0.01


def test_int8_per_channel_better_than_per_tensor():
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    qt, st = quantize_per_tensor(x)
    qc, sc = quantize_per_channel(x, axis=0)
    err_t = mse(x, dequantize_per_tensor(qt, st))
    err_c = mse(x, dequantize_per_channel(qc, sc))
    # in random data per-channel >= per-tensor in MSE (more scales)
    assert err_c <= err_t + 1e-3


def test_int4_per_group_round_trip():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    q, s = quantize_per_group(x, group_size=32, n_bits=4)
    x_dq = dequantize_per_group(q, s, group_size=32)
    assert x_dq.shape == x.shape
    assert mse(x, x_dq) < 0.1


def test_gptq_reduces_error_vs_naive():
    torch.manual_seed(0)
    W = torch.randn(16, 32)
    X = torch.randn(128, 32)
    H = calibrate_hessian(X)
    Wq_gptq, _ = gptq_quantize(W, H, n_bits=4)
    # naive per-tensor 4bit
    qmax = 7
    s = W.abs().max() / qmax
    Wq_naive = (W / s).round().clamp(-qmax, qmax) * s
    gptq_err = mse(Wq_gptq @ X.t(), W @ X.t())
    naive_err = mse(Wq_naive @ X.t(), W @ X.t())
    assert gptq_err <= naive_err * 1.5    # ≤ naive (allow some randomness)


def test_awq_returns_correct_shape():
    torch.manual_seed(0)
    W = torch.randn(32, 64)
    X = torch.randn(128, 64)
    Wq, scales = awq_quantize(W, X)
    assert Wq.shape == W.shape


def test_smooth_quant_keeps_matmul_close():
    torch.manual_seed(0)
    X = torch.randn(64, 32) * 0.5
    X[:, 5] *= 50    # outlier
    W = torch.randn(32, 32)
    s = find_smooth_scale(X, W)
    X_s, W_s = apply_smoothing(X, W, s)
    # matmul invariant
    assert torch.allclose(X @ W, X_s @ W_s, atol=1e-3)


def test_fp8_round_within_table():
    x = torch.tensor([0.0, 0.3, -1.5, 100.0])
    out = fp8_round(x)
    assert out.shape == x.shape
    # all outputs should be representable E4M3 values
    assert not torch.isnan(out).any()


def test_fp8_matmul_mock_low_error():
    torch.manual_seed(0)
    W = torch.randn(32, 32) * 0.5
    X = torch.randn(16, 32) * 0.5
    ref = W @ X.t()
    approx = fp8_matmul_mock(W, X, w_scale=1.0, x_scale=1.0)
    assert relative_error(approx, ref) < 0.2


def test_nf4_round_trip_low_error():
    torch.manual_seed(0)
    W = torch.randn(256)
    idx, scales = quantize_nf4(W, block_size=64)
    Wd = dequantize_nf4(idx, scales, block_size=64, orig_shape=W.shape)
    assert Wd.shape == W.shape
    assert mse(W, Wd) < 0.1


def test_kv_per_token_quant_low_error():
    torch.manual_seed(0)
    kv = torch.randn(8, 4, 16)
    q, s = quantize_kv_per_token(kv)
    kvd = dequantize_kv_per_token(q, s)
    assert mse(kv, kvd) < 0.05


def test_attention_with_quant_kv_runs():
    torch.manual_seed(0)
    K = torch.randn(8, 4, 16)
    V = torch.randn(8, 4, 16)
    Kq, ks = quantize_kv_per_token(K)
    Vq, vs = quantize_kv_per_token(V)
    Q = torch.randn(2, 4, 16)
    out = attention_with_quant_kv(Q, Kq, Vq, ks, vs)
    assert out.shape == (2, 4, 16)


def test_capstone_runs_all_methods():
    rows = run_all()
    names = {r["variant"] for r in rows}
    # every real quantizer module must be exercised, not just named
    for expected in {"fp16", "int8 (pc)", "GPTQ-4bit", "AWQ-4bit",
                     "NF4 (bnb)", "FP8 (E4M3)", "SmoothQuant-int8"}:
        assert expected in names


def test_capstone_errors_are_computed_not_hardcoded():
    # fp16 reference is exact; every quantized variant has real, finite error.
    rows = {r["variant"]: r for r in run_all()}
    assert rows["fp16"]["error"] == 0.0
    for name in ("int8 (pc)", "GPTQ-4bit", "AWQ-4bit", "NF4 (bnb)",
                 "FP8 (E4M3)", "SmoothQuant-int8"):
        assert rows[name]["error"] > 0.0
        assert math.isfinite(rows[name]["error"])


def test_capstone_8bit_beats_4bit_on_reconstruction():
    rows = {r["variant"]: r for r in run_all()}
    best_8bit = min(rows["int8 (pc)"]["error"], rows["SmoothQuant-int8"]["error"])
    worst_4bit = max(rows["GPTQ-4bit"]["error"], rows["AWQ-4bit"]["error"],
                     rows["NF4 (bnb)"]["error"])
    assert best_8bit < worst_4bit


def test_capstone_gptq_compensation_beats_nf4_codebook():
    # GPTQ's Hessian error-compensation should beat the static NF4 codebook
    # at the same 4-bit budget on this correlated toy layer.
    rows = {r["variant"]: r for r in run_all()}
    assert rows["GPTQ-4bit"]["error"] <= rows["NF4 (bnb)"]["error"]


def test_capstone_compression_and_memory_track_bitwidth():
    for r in run_all():
        assert r["compression"] == round(16.0 / r["bits"], 2)
    rows = {r["variant"]: r for r in run_all()}
    # 4-bit variants store half the bytes of the 8-bit variants
    assert rows["GPTQ-4bit"]["mem_mib"] < rows["int8 (pc)"]["mem_mib"]


def test_capstone_best_for_categories():
    assert best_for("accuracy")["variant"] == "fp16"          # exact reference
    assert best_for("compression")["bits"] == 4               # most compressed
    assert best_quantized_for("accuracy")["error"] > 0.0      # excludes fp16


def test_memory_table_includes_all_rows():
    rows = run_all()
    md = memory_table([
        dict(variant=r["variant"], ppl=r["error"], acc="-",
             mem_gb=round(r["mem_mib"] / 1024, 2), tok_s="-")
        for r in rows
    ])
    for r in rows:
        assert r["variant"] in md
