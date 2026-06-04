"""Long Context 环境自检."""
from __future__ import annotations
import sys


def part_a():
    print("\n=== Part A: 基础 ===")
    ok = True
    for m in ["torch", "einops", "transformers"]:
        try: __import__(m); print(f"  [OK] {m}")
        except ImportError as e: print(f"  [FAIL] {m}: {e}"); ok = False
    for m in ["flash_attn", "ring_flash_attention"]:
        try: __import__(m); print(f"  [OK-opt] {m}")
        except ImportError: print(f"  [SKIP] {m}")
    return ok


def part_b():
    import torch
    print("\n=== Part B: GPU ===")
    if torch.cuda.is_available():
        print(f"  [OK] {torch.cuda.get_device_name(0)}")
    else:
        print("  [SKIP] no CUDA")
    return True


def part_c():
    print("\n=== Part C: YaRN scaling smoke ===")
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
        from rope_yarn import yarn_cos_sin
        cos, sin = yarn_cos_sin(t=128, dim=64, base=10000.0, factor=4.0,
                                 original_max_pos=2048)
        print(f"  [OK] YaRN cos {cos.shape}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    res = [part_a(), part_b(), part_c()]
    for n, ok in zip("ABC", res):
        print(f"  Part {n}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(res) else 1)
