"""SSM/Hybrid 环境自检."""
from __future__ import annotations
import sys


def part_a():
    print("\n=== Part A: 基础 ===")
    ok = True
    for m in ["torch", "einops"]:
        try: __import__(m); print(f"  [OK] {m}")
        except ImportError as e: print(f"  [FAIL] {m}: {e}"); ok = False
    for m in ["mamba_ssm", "causal_conv1d", "rwkv"]:
        try: __import__(m); print(f"  [OK-opt] {m}")
        except ImportError: print(f"  [SKIP] {m} (WSL2 only)")
    return ok


def part_b():
    print("\n=== Part B: GPU ===")
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] {torch.cuda.get_device_name(0)}")
    else:
        print("  [SKIP] no CUDA")
    return True


def part_c():
    print("\n=== Part C: Mamba forward smoke ===")
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
        from mamba_block import MambaBlock
        import torch
        m = MambaBlock(d_model=32, d_state=8, d_conv=4)
        x = torch.randn(1, 16, 32)
        y = m(x)
        print(f"  [OK] Mamba out {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    res = [part_a(), part_b(), part_c()]
    for n, ok in zip("ABC", res):
        print(f"  Part {n}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(res) else 1)
