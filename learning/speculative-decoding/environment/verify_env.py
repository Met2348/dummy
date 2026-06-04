"""Env check for speculative decoding workshop."""
import sys


def main():
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print("verify_env OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
