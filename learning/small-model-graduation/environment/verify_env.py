"""Topic 8 small-model-graduation env check."""
import sys
import importlib

def check(m):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  [OK] {m:18s} {v}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {m}: {e}")
        return False

ok = True
for m in ["torch", "transformers", "datasets", "accelerate", "peft"]:
    ok &= check(m)
import torch
print(f"  CUDA: {torch.cuda.is_available()}")
sys.exit(0 if ok else 1)
