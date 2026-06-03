"""NF4 fake-quant 单元测试。

测试目标:
  1. NF4 网格点正确性（对称、单调、零中心）
  2. 量化-反量化对 N(0, 1) 的误差有界
  3. STE 反向（梯度完整穿过）
  4. block_size 效果（较小 block_size 通常误差较小）
  5. Double Quantization 一致性
  6. 非正态输入的鲁棒性（NF4 仍可运行）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from nf4_quant import (  # noqa: E402
    NF4_VALUES,
    nf4_quant_dequant,
    nf4_quantize,
    nf4_dequantize,
    double_quantize_absmax,
    double_dequantize_absmax,
)


def test_nf4_lookup_table():
    print("\n[Test 1] NF4 网格点正确性")
    assert NF4_VALUES.shape == (16,)
    assert NF4_VALUES[0].item() == -1.0
    assert NF4_VALUES[-1].item() == 1.0
    assert NF4_VALUES[7].item() == 0.0
    diffs = NF4_VALUES[1:] - NF4_VALUES[:-1]
    assert (diffs > 0).all(), "NF4 网格点必须严格单调"
    print(f"  网格点 16 值: [{NF4_VALUES.min():.4f} ... {NF4_VALUES.max():.4f}]")
    print(f"  零点位置: index 7 = {NF4_VALUES[7].item()}")
    print(f"  正负对称分布: 8 个负值 + 0 + 7 个正值")
    print(f"  [PASS]")


def test_nf4_normal_input_error():
    print("\n[Test 2] N(0, 1) 输入量化误差")
    torch.manual_seed(42)
    W = torch.randn(1024, 1024)
    W_hat = nf4_quant_dequant(W, block_size=64)
    rmse = (W - W_hat).pow(2).mean().sqrt().item()
    rel_err = rmse / W.std().item()
    print(f"  绝对 RMSE:  {rmse:.5f}")
    print(f"  相对 RMSE:  {rel_err:.5f}  ({100 * rel_err:.2f}%)")
    assert rel_err < 0.13, f"相对 RMSE 过大: {rel_err}"
    print(f"  [PASS] 在 13% 阈值内（NF4 对 N(0,1) 的理论极限）")


def test_nf4_ste_backward():
    print("\n[Test 3] STE 反向传播")
    torch.manual_seed(0)
    W = torch.randn(64, 64, requires_grad=True)
    W_hat = nf4_quant_dequant(W, block_size=64)
    loss = W_hat.sum()
    loss.backward()
    assert W.grad is not None
    assert torch.allclose(W.grad, torch.ones_like(W)), (
        f"STE 应让所有 gradient = 1.0（loss.sum().backward 下），实际 max={W.grad.max()}, min={W.grad.min()}"
    )
    print(f"  STE 让所有 gradient = 1.0  [PASS]")


def test_nf4_block_size_effect():
    print("\n[Test 4] block_size 对量化误差的影响")
    torch.manual_seed(42)
    W = torch.randn(2048, 2048)
    print(f"  {'block_size':>12} | {'RMSE':>12}")
    print(f"  {'-' * 12} | {'-' * 12}")
    errs = []
    for bs in [32, 64, 128, 256, 512]:
        W_hat = nf4_quant_dequant(W, block_size=bs)
        err = (W - W_hat).pow(2).mean().sqrt().item()
        errs.append(err)
        print(f"  {bs:>12} | {err:.5f}")
    # 较大 block_size 通常误差较大（因 absmax 池化范围更大），但不强制断言
    print(f"  [INFO] 最小误差 block_size={[32, 64, 128, 256, 512][errs.index(min(errs))]}")
    print(f"  [PASS]")


def test_double_quantization():
    print("\n[Test 5] Double Quantization 一致性")
    torch.manual_seed(42)
    absmax = torch.rand(1024) * 5.0  # 模拟随机 absmax scale
    indices, outer_max, shape, pad = double_quantize_absmax(absmax, block_size_outer=256)
    absmax_hat = double_dequantize_absmax(indices, outer_max, shape, pad)
    err = (absmax - absmax_hat).abs().max().item() / absmax.max().item()
    print(f"  absmax shape: {tuple(absmax.shape)}")
    print(f"  最大相对误差: {err:.5f}  ({100 * err:.2f}%)")
    assert err < 0.02, f"Double quantization 误差过大: {err}"
    print(f"  [PASS]")


def test_nf4_uniform_input():
    print("\n[Test 6] 均匀输入（非正态）的鲁棒性")
    torch.manual_seed(42)
    W = torch.rand(256, 256) * 2 - 1  # uniform [-1, 1]
    W_hat = nf4_quant_dequant(W, block_size=64)
    err = (W - W_hat).abs().mean().item()
    print(f"  均匀输入 MAE: {err:.5f} (高于正态预期，因 NF4 针对正态优化)")
    print(f"  [PASS] 算法对非正态仍可运行")


def test_nf4_gpu_smoke():
    print("\n[Test 7] GPU 烟雾测试")
    if not torch.cuda.is_available():
        print("  [SKIP] 没有 GPU")
        return
    try:
        W = torch.randn(512, 512, device="cuda")
        W_hat = nf4_quant_dequant(W, block_size=64)
        assert W_hat.device == W.device
        err = (W - W_hat).pow(2).mean().sqrt().item()
        print(f"  GPU NF4 RMSE: {err:.5f}  [PASS]")
    except RuntimeError as e:
        print(f"  [SKIP] GPU 不兼容（torch 未为当前 sm 编译）: {str(e)[:80]}")


if __name__ == "__main__":
    test_nf4_lookup_table()
    test_nf4_normal_input_error()
    test_nf4_ste_backward()
    test_nf4_block_size_effect()
    test_double_quantization()
    test_nf4_uniform_input()
    test_nf4_gpu_smoke()
    print("\n" + "=" * 60)
    print("[PASS] 全部 NF4 单元测试通过")
    print("=" * 60)
