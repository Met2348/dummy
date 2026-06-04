"""Env check for Module 5 graduation capstone."""
import sys


def main():
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print("verify_env OK")


if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"FAIL: {e}"); sys.exit(1)
