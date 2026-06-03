"""Transformer Deep 环境自检 — 三段式.

Part A: torch + einops + transformers import
Part B: GPU + sm 检测（5090 是 sm_120）
Part C: 简化 GPT-mini 单 forward smoke
"""
from __future__ import annotations

import sys


def part_a() -> bool:
    print("\n=== Part A: 基础 import ===")
    ok = True
    for mod in ["torch", "einops", "transformers"]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            print(f"  [OK] {mod}={v}")
        except ImportError as e:
            print(f"  [FAIL] {mod}: {e}")
            ok = False
    # 可选库
    for mod in ["flash_attn", "triton"]:
        try:
            __import__(mod)
            print(f"  [OK-optional] {mod}")
        except ImportError:
            print(f"  [SKIP] {mod} (optional)")
    return ok


def part_b() -> bool:
    print("\n=== Part B: GPU + sm ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [FAIL] CUDA 不可用")
            return False
        n = torch.cuda.device_count()
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"  [OK] GPU {i}: {name}  sm_{cap[0]}{cap[1]}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def part_c() -> bool:
    print("\n=== Part C: GPT-mini single forward smoke ===")
    try:
        import torch
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
        from gpt_mini import GPTMini, GPTMiniConfig
        cfg = GPTMiniConfig(vocab_size=512, n_layer=2, n_head=4, n_kv=2,
                            d_model=64, max_seq=128)
        model = GPTMini(cfg).cuda()
        x = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size), logits.shape
        print(f"  [OK] forward logits {logits.shape}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    results = [part_a(), part_b(), part_c()]
    print("\n=== 汇总 ===")
    for n, ok in zip("ABC", results):
        print(f"  Part {n}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results) else 1)
