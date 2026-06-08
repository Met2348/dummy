"""Shared memory + bank conflicts model."""
from __future__ import annotations

N_BANKS = 32
BANK_WIDTH_BYTES = 4


def access_bank(byte_offset: int) -> int:
    """Which bank does this byte offset map to?"""
    return (byte_offset // BANK_WIDTH_BYTES) % N_BANKS


def count_conflicts(accesses_per_lane: list[int]) -> int:
    """Given 32 byte offsets (one per lane), how many bank conflicts?

    Conflict = multiple lanes touching the same bank with different addresses.
    Broadcast (same word) does not conflict.
    """
    assert len(accesses_per_lane) == 32
    bank_to_addrs: dict[int, set[int]] = {}
    for offset in accesses_per_lane:
        bank = access_bank(offset)
        bank_to_addrs.setdefault(bank, set()).add(offset)
    return sum(len(addrs) - 1 for addrs in bank_to_addrs.values() if len(addrs) > 1)


def stride_access(stride_words: int) -> list[int]:
    return [lane * stride_words * BANK_WIDTH_BYTES for lane in range(32)]


def _self_test() -> None:
    # Stride 1: consecutive 4-byte words, 0 conflicts.
    assert count_conflicts(stride_access(1)) == 0
    # Stride 32: every lane hits the same bank, giving a 31-way conflict.
    assert count_conflicts(stride_access(32)) == 31
    # Stride 2: two lanes collide on each used bank.
    conf_s2 = count_conflicts(stride_access(2))
    assert conf_s2 > 0, "stride 2 conflicts"
    # Broadcast: all lanes read the same word, so there are 0 conflicts.
    assert count_conflicts([4] * 32) == 0
    print(f"[OK] shared_memory (stride2 conflicts {conf_s2})")


if __name__ == "__main__":
    _self_test()
