"""LoRA 家族专题环境自检。

三段式：
  Part A: 基础导入（必须全 PASS）
  Part B: GPU 检测（可选）
  Part C: bitsandbytes（GPU + 真量化，QLoRA 选做 cell 用）
"""
from __future__ import annotations

import importlib
import sys


def _parse_version(s: str) -> tuple:
    """字符串版本号解析为 tuple，用于正确比较。"""
    parts = []
    for x in s.split("."):
        # 取数字前缀（如 "0+cu130" → 0）
        digits = ""
        for c in x:
            if c.isdigit():
                digits += c
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def part_a_basic() -> bool:
    print("=" * 60)
    print("Part A: 基础包导入（必须全 PASS）")
    print("=" * 60)
    required = [
        ("torch", "2.5"),
        ("transformers", "5.0"),
        ("peft", "0.13"),
        ("scipy", "1.10"),
        ("matplotlib", "3.7"),
        ("numpy", "1.24"),
        ("datasets", "2.14"),
        ("accelerate", "0.30"),
    ]
    all_pass = True
    for name, min_ver in required:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            ok = _parse_version(ver) >= _parse_version(min_ver)
            tag = "[OK]" if ok else "[OLD]"
            print(f"  {tag} {name}: {ver} (>= {min_ver} required)")
            if not ok:
                all_pass = False
        except ImportError as e:
            print(f"  [MISSING] {name}: {e}")
            all_pass = False
    return all_pass


def part_b_gpu() -> bool:
    print("\n" + "=" * 60)
    print("Part B: GPU 检测（CUDA torch + 显存）")
    print("=" * 60)
    import torch

    if not torch.cuda.is_available():
        print(f"  [CPU ONLY] torch={torch.__version__}, CUDA built={torch.version.cuda}")
        print("  注: QLoRA / LoftQ 的 GPU 选做 cell 将自动 SKIP")
        return False
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    print(f"  [INFO] GPU: {name}")
    print(f"  [INFO] VRAM: {mem:.1f} GB")
    print(f"  [INFO] Compute capability: sm_{cap[0]}{cap[1]}")
    print(f"  [INFO] torch CUDA: {torch.version.cuda}")
    # 简单 GEMM 测试，包 try/except 处理 sm_xx 编译错配
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"  [OK] GEMM smoke test: 1024x1024 matmul OK, sum={c.sum().item():.2f}")
        return True
    except RuntimeError as e:
        print(f"  [FAIL] GEMM 失败: {type(e).__name__}")
        print(f"  [FAIL] 错误: {str(e)[:150]}")
        print(f"  [HINT] torch 未为 sm_{cap[0]}{cap[1]} 编译；GPU cell 将 SKIP")
        return False


def part_c_bitsandbytes() -> bool:
    print("\n" + "=" * 60)
    print("Part C: bitsandbytes（GPU + 真 NF4，QLoRA 选做 cell 用）")
    print("=" * 60)
    try:
        import bitsandbytes as bnb
        print(f"  [OK] bitsandbytes: {bnb.__version__}")
    except ImportError:
        print("  [MISSING] bitsandbytes（QLoRA 选做 cell 会 SKIP）")
        return False
    import torch
    if not torch.cuda.is_available():
        print("  [SKIP] 没有 GPU，bitsandbytes NF4 无法测试")
        return False
    try:
        # 一次 NF4 quantize/dequantize 调用
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        q, s = bnb.functional.quantize_nf4(x)
        x_hat = bnb.functional.dequantize_nf4(q, s)
        err = (x - x_hat).abs().mean().item()
        print(f"  [OK] bnb NF4 quant-dequant 平均误差: {err:.4f}")
        return True
    except Exception as e:
        print(f"  [FAIL] bnb NF4 调用失败: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print(f"Python: {sys.version.split()[0]}\n")
    a = part_a_basic()
    b = part_b_gpu()
    c = part_c_bitsandbytes()
    print("\n" + "=" * 60)
    print(f"Part A (基础):           {'PASS' if a else 'FAIL'}")
    print(f"Part B (GPU):            {'PASS' if b else 'SKIP'}")
    print(f"Part C (bitsandbytes):   {'PASS' if c else 'SKIP'}")
    print("=" * 60)
    if not a:
        print("\n[ERROR] Part A 基础包未就绪，请先装 requirements.txt")
        return 1
    if not b:
        print("\n[NOTE] 没有 GPU。主线代码仍可在 CPU 上跑；QLoRA/LoftQ GPU 选做 cell 会 SKIP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
