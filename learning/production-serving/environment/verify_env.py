"""Env check for production serving workshop."""
import sys


def main():
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    for lib in ("fastapi", "uvicorn", "pydantic", "prometheus_client"):
        try:
            mod = __import__(lib)
            print(f"  {lib} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  {lib} NOT installed (lecture-only ok)")
    print("verify_env OK")


if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"FAIL: {e}"); sys.exit(1)
