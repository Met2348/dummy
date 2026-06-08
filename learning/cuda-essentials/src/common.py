"""Mock CUDA execution model: grid, block, warp, thread."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Thread:
    tid_x: int
    tid_y: int = 0
    tid_z: int = 0


@dataclass
class Block:
    bid_x: int
    dim: tuple[int, int, int]
    threads: list[Thread] = field(default_factory=list)

    def __post_init__(self):
        if not self.threads:
            for z in range(self.dim[2]):
                for y in range(self.dim[1]):
                    for x in range(self.dim[0]):
                        self.threads.append(Thread(x, y, z))

    def n_threads(self) -> int:
        return self.dim[0] * self.dim[1] * self.dim[2]

    def n_warps(self) -> int:
        return (self.n_threads() + 31) // 32


@dataclass
class Grid:
    dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]

    def n_blocks(self) -> int:
        return self.dim[0] * self.dim[1] * self.dim[2]

    def total_threads(self) -> int:
        return self.n_blocks() * (self.block_dim[0] * self.block_dim[1] * self.block_dim[2])


def launch_config(problem_size: int, threads_per_block: int = 256) -> Grid:
    """Standard 1D launch config sized to problem."""
    n_blocks = (problem_size + threads_per_block - 1) // threads_per_block
    return Grid((n_blocks, 1, 1), (threads_per_block, 1, 1))


def _self_test() -> None:
    g = launch_config(10_000, 256)
    assert g.n_blocks() == 40                       # ceil(10000/256) = 40
    assert g.total_threads() == 10240
    b = Block(0, (256, 1, 1))
    assert b.n_threads() == 256
    assert b.n_warps() == 8
    # 100 threads gives 4 warps, with the last warp partially used.
    b2 = Block(0, (100, 1, 1))
    assert b2.n_warps() == 4
    print("[OK] cuda_essentials.common")


if __name__ == "__main__":
    _self_test()
