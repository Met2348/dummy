"""Adapter Tuning Family 环境自检 — 三段式。

Part A: 基础（torch / transformers / peft / adapters / 其它）
Part B: GPU + sm_120 (Blackwell)
Part C: adapters 库 GPT-2 smoke test
"""
from __future__ import annotations

import sys


def _parse_version(v: str) -> tuple:
    """'2.13.0.dev20260602+cu130' → (2, 13, 0)"""
    parts = v.split("+")[0].split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p.split("dev")[0] or "0"))
        except ValueError:
            break
    return tuple(nums)


def part_a_basic() -> bool:
    print("\n=== Part A: 基础 ===")
    targets = {
        "torch": "2.5",
        "transformers": "4.55",
        "adapters": "1.3",
        "peft": "0.13",
        "bitsandbytes": "0.43",
        "scipy": "1.10",
        "matplotlib": "3.7",
        "numpy": "1.24",
        "datasets": "2.14",
        "accelerate": "0.30",
    }
    ok = True
    for name, min_v in targets.items():
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            if _parse_version(ver) >= _parse_version(min_v):
                print(f"  [OK]   {name:<14} {ver}  (>= {min_v})")
            else:
                print(f"  [WARN] {name:<14} {ver}  (< {min_v})")
                ok = False
        except ImportError:
            print(f"  [FAIL] {name:<14} not installed")
            ok = False
    print(f"\nPart A: {'PASS' if ok else 'FAIL'}")
    return ok


def part_b_gpu() -> bool:
    print("\n=== Part B: GPU ===")
    try:
        import torch
    except ImportError:
        print("  torch not installed, SKIP")
        return True
    if not torch.cuda.is_available():
        print("  No CUDA GPU, SKIP（CPU 仍可跑 minimal 代码）")
        return True
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"  compute: {sm}")
    if sm == "sm_120":
        print("  Blackwell GPU 需要 cu130+ torch")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"  GEMM smoke: [OK] (output norm = {y.norm().item():.2e})")
        print("\nPart B: PASS")
        return True
    except RuntimeError as e:
        print(f"  GEMM failed: {e}")
        print("\nPart B: FAIL")
        return False


def part_c_adapters() -> bool:
    print("\n=== Part C: adapters 库 smoke test ===")
    try:
        from adapters import AutoAdapterModel, AdapterConfig
    except ImportError:
        print("  adapters not installed, FAIL")
        return False
    try:
        model = AutoAdapterModel.from_pretrained("gpt2")
        config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        model.add_adapter("test", config=config)
        model.train_adapter("test")
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Pfeiffer adapter trainable: {n_train:,}")
        print("  GPT-2 + adapters: [OK]")
        print("\nPart C: PASS")
        return True
    except Exception as e:
        print(f"  smoke test failed: {e}")
        print("\nPart C: FAIL")
        return False


def main() -> int:
    print("Adapter Tuning Family 环境自检")
    print("=" * 50)
    a = part_a_basic()
    b = part_b_gpu()
    c = part_c_adapters()
    print("\n" + "=" * 50)
    print(f"总结: A={'PASS' if a else 'FAIL'} B={'PASS' if b else 'SKIP/FAIL'} C={'PASS' if c else 'FAIL'}")
    return 0 if (a and c) else 1


if __name__ == "__main__":
    sys.exit(main())
