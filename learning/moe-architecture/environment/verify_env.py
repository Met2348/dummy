"""MoE Architecture 环境自检."""
from __future__ import annotations

import sys


def part_a() -> bool:
    print("\n=== Part A: 基础 import ===")
    ok = True
    for mod in ["torch", "einops"]:
        try:
            __import__(mod); print(f"  [OK] {mod}")
        except ImportError as e:
            print(f"  [FAIL] {mod}: {e}"); ok = False
    for mod in ["megablocks", "deepspeed"]:
        try:
            __import__(mod); print(f"  [OK-opt] {mod}")
        except ImportError:
            print(f"  [SKIP] {mod} (WSL2 only)")
    return ok


def part_b() -> bool:
    print("\n=== Part B: GPU ===")
    import torch
    if not torch.cuda.is_available():
        print("  [SKIP] no CUDA")
        return True
    print(f"  [OK] {torch.cuda.get_device_name(0)}")
    return True


def part_c() -> bool:
    print("\n=== Part C: top-2 MoE forward smoke ===")
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
        from moe_layer_naive import MoELayer
        import torch
        layer = MoELayer(d_model=32, n_experts=4, top_k=2, d_ff=64)
        x = torch.randn(2, 8, 32)
        out, aux = layer(x)
        assert out.shape == x.shape, out.shape
        print(f"  [OK] MoE out {tuple(out.shape)} aux={aux.item():.4f}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    res = [part_a(), part_b(), part_c()]
    print("\n=== 汇总 ===")
    for n, ok in zip("ABC", res):
        print(f"  Part {n}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(res) else 1)
