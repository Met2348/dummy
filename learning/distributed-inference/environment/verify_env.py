"""Env check for distributed inference workshop."""
import sys


def main():
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device_count={torch.cuda.device_count()}")
    try:
        import ray
        print(f"ray {ray.__version__}")
    except ImportError:
        print("ray NOT installed (mock-only flow ok)")
    print("verify_env OK")


if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"FAIL: {e}"); sys.exit(1)
