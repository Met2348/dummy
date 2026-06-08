"""Global memory coalescing model: 128-byte sector."""
from __future__ import annotations

SECTOR_BYTES = 128
DTYPE_BYTES = 4


def n_sectors(addresses: list[int]) -> int:
    """How many 128B sectors are touched by these addresses?"""
    sectors = set(a // SECTOR_BYTES for a in addresses)
    return len(sectors)


def coalesced_load(base: int, n_lanes: int = 32) -> list[int]:
    """Lane i reads base + i*4, the ideal pattern."""
    return [base + i * DTYPE_BYTES for i in range(n_lanes)]


def strided_load(base: int, stride_elem: int, n_lanes: int = 32) -> list[int]:
    """Lane i reads base + i*stride*4, a poor pattern when stride is large."""
    return [base + i * stride_elem * DTYPE_BYTES for i in range(n_lanes)]


def efficiency(addresses: list[int]) -> float:
    """Bytes_used / bytes_loaded. Ideal coalesced = 1.0."""
    bytes_used = len(addresses) * DTYPE_BYTES
    bytes_loaded = n_sectors(addresses) * SECTOR_BYTES
    return bytes_used / bytes_loaded


def _self_test() -> None:
    # Coalesced: 32 lanes * 4 byte = 128 B, so one sector.
    addrs = coalesced_load(0)
    assert n_sectors(addrs) == 1
    assert efficiency(addrs) == 1.0
    # Stride 8: lanes 0-3 share sector 0 (addrs 0,32,64,96), 4-7 share sector 1, ...
    # 32 lanes * stride 8 * 4 B = 1024 B / 128 B = 8 sectors.
    addrs_s8 = strided_load(0, 8)
    assert n_sectors(addrs_s8) == 8, n_sectors(addrs_s8)
    # Stride 32: each lane lands in its own sector, so efficiency is 1/32.
    addrs_s32 = strided_load(0, 32)
    assert n_sectors(addrs_s32) == 32, n_sectors(addrs_s32)
    assert abs(efficiency(addrs_s32) - (1.0 / 32.0)) < 1e-6
    # Stride 2: 32 * 8B = 256 B, so 2 sectors and efficiency 0.5.
    addrs_s2 = strided_load(0, 2)
    assert n_sectors(addrs_s2) == 2
    assert efficiency(addrs_s2) == 0.5
    print(f"[OK] coalescing (stride32 eff {efficiency(addrs_s32):.4f})")


if __name__ == "__main__":
    _self_test()
