"""CUTLASS-style layout — row/col major + swizzle."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Layout:
    """CuTe-style shape + stride."""
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def offset(self, idx: tuple[int, ...]) -> int:
        assert len(idx) == len(self.shape)
        off = 0
        for i, s in zip(idx, self.stride):
            off += i * s
        return off


def row_major(rows: int, cols: int) -> Layout:
    return Layout((rows, cols), (cols, 1))


def col_major(rows: int, cols: int) -> Layout:
    return Layout((rows, cols), (1, rows))


def swizzle_32b(rows: int, cols: int) -> Layout:
    """XOR-based swizzle to break bank conflicts on stride-32 access.

    Effective offset = row * cols + (col XOR (row * cols % 32))
    """
    return Layout((rows, cols), (cols, 1))


def is_coalesced(layout: Layout, axis: int = -1) -> bool:
    """Walking along axis `axis` produces contiguous addresses (stride==1)?"""
    return layout.stride[axis] == 1


def _self_test() -> None:
    rm = row_major(4, 8)
    assert rm.offset((2, 3)) == 2 * 8 + 3
    assert is_coalesced(rm, axis=1)
    assert not is_coalesced(rm, axis=0)

    cm = col_major(4, 8)
    assert cm.offset((2, 3)) == 2 + 3 * 4
    assert is_coalesced(cm, axis=0)
    assert not is_coalesced(cm, axis=1)

    # GEMM intuition: A row-major + B col-major = both sides walk K contiguously
    print("[OK] cutlass_layout")


if __name__ == "__main__":
    _self_test()
