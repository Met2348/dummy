"""ring-flash-attention 库 wrapper (多 GPU 才有效)."""
from __future__ import annotations


def has_lib() -> bool:
    try:
        import ring_flash_attention  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    print("ring-flash-attention available:", has_lib())
    if not has_lib():
        print("[SKIP] 仅 Linux 多 GPU 可用")
