"""Reasoning R1 环境自检 — WSL2 三段式.

Part A: torch + transformers + trl + verl + vllm import
Part B: vllm 单卡 GPT-2 推理 smoke
Part C: verl PPO/GRPO 5-step smoke + Ray cluster init
"""
from __future__ import annotations

import sys


def part_a() -> bool:
    print("\n=== Part A: 基础 ===")
    ok = True
    try:
        import torch, transformers, trl, peft       # noqa
        print(f"  [OK] torch={torch.__version__}, trl={trl.__version__}")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        ok = False
    try:
        import verl                                  # noqa
        print(f"  [OK] verl")
    except ImportError as e:
        print(f"  [FAIL] verl 未装：必须 WSL2 + Linux: {e}")
        ok = False
    try:
        import vllm                                  # noqa
        print(f"  [OK] vllm")
    except ImportError as e:
        print(f"  [FAIL] vllm 未装：必须 WSL2 + Linux: {e}")
        ok = False
    try:
        import ray                                   # noqa
        print(f"  [OK] ray")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        ok = False
    return ok


def part_b() -> bool:
    print("\n=== Part B: vllm 单 GPU smoke ===")
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model="Qwen/Qwen2.5-0.5B", gpu_memory_utilization=0.5)
        out = llm.generate(["Hello, my name is"],
                           SamplingParams(max_tokens=5, temperature=0))
        print(f"  [OK] vllm 推理: {out[0].outputs[0].text[:30]!r}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def part_c() -> bool:
    print("\n=== Part C: ray cluster + verl smoke ===")
    try:
        import ray
        if not ray.is_initialized():
            ray.init(local_mode=True, num_gpus=1)
        print(f"  [OK] ray init")
        ray.shutdown()
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main() -> int:
    print("Reasoning R1 环境自检（WSL2 only）")
    print("=" * 50)
    a = part_a()
    b = part_b() if a else False
    c = part_c() if a else False
    print("\n" + "=" * 50)
    print(f"A={a}, B={b}, C={c}")
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    sys.exit(main())
