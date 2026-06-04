"""3-part env check: torch+cuda, vllm/triton, helper libs."""
import sys


def part_a_basic():
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    cap = torch.cuda.get_device_capability()
    print(f"[A] torch {torch.__version__} CUDA {torch.version.cuda} cap={cap}")


def part_b_engine():
    try:
        import vllm
        print(f"[B] vllm {vllm.__version__}")
    except ImportError:
        print("[B] vllm NOT installed (ok for minimal-only flow)")
    try:
        import triton
        print(f"[B] triton {triton.__version__}")
    except ImportError:
        print("[B] triton NOT installed")
    try:
        import flash_attn
        print(f"[B] flash-attn {flash_attn.__version__}")
    except ImportError:
        print("[B] flash-attn NOT installed (slow path ok)")


def part_c_helpers():
    import transformers
    import numpy
    print(f"[C] transformers {transformers.__version__} numpy {numpy.__version__}")


if __name__ == "__main__":
    parts = [part_a_basic, part_b_engine, part_c_helpers]
    for p in parts:
        try:
            p()
        except Exception as e:
            print(f"FAIL {p.__name__}: {e}")
            sys.exit(1)
    print("\nverify_env OK")
