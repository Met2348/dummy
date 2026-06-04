"""Topic 7 pretraining-recipe env check."""
import sys
import importlib

def check(m):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  [OK] {m:20s} {v}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {m:20s} {e}")
        return False


print("=== Part A: 基础 ===")
ok = True
for m in ["torch", "transformers", "datasets", "accelerate", "tokenizers"]:
    ok &= check(m)

print("\n=== Part B: GPU ===")
import torch
print(f"  CUDA: {torch.cuda.is_available()}")

print("\n=== Part C: 可选 ===")
for m in ["tiktoken", "sentencepiece"]:
    check(m)

sys.exit(0 if ok else 1)
