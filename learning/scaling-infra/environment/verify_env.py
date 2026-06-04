"""Topic 6 scaling-infra env check."""
import sys
import importlib

def check(mod):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [OK] {mod:20s} {v}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {mod:20s} {e}")
        return False


print("=== Part A: 基础 ===")
ok = True
for m in ["torch", "transformers", "accelerate"]:
    ok &= check(m)

print("\n=== Part B: GPU ===")
import torch
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n=== Part C: 推理/训练库 (optional, 多在 WSL2) ===")
for m in ["bitsandbytes", "deepspeed", "vllm"]:
    check(m)

print("\nALL OK" if ok else "\nMISSING required packages")
sys.exit(0 if ok else 1)
