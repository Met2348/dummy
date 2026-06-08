"""mamba-ssm library wrapper - compare with naive implementation."""
from __future__ import annotations

import torch


def has_mamba_lib() -> bool:
    try:
        import mamba_ssm  # noqa
        return True
    except ImportError:
        return False


def lib_demo():
    if not has_mamba_lib():
        print("[SKIP] mamba-ssm not installed (Linux/WSL2 only)")
        return
    from mamba_ssm import Mamba
    m = Mamba(d_model=64, d_state=16, d_conv=4).cuda()
    x = torch.randn(1, 32, 64, device="cuda")
    y = m(x)
    print(f"mamba-ssm lib out {y.shape}")


if __name__ == "__main__":
    print("mamba-ssm available:", has_mamba_lib())
    lib_demo()
