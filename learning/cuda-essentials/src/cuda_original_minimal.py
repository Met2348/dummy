"""CUDA Programming Guide concepts as a small executable model."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

WARP_SIZE = 32


@dataclass(frozen=True)
class Dim3:
    x: int
    y: int = 1
    z: int = 1

    def volume(self) -> int:
        return self.x * self.y * self.z


@dataclass(frozen=True)
class LaunchConfig:
    grid: Dim3
    block: Dim3
    shared_mem_bytes: int = 0
    stream: int = 0

    def total_threads(self) -> int:
        return self.grid.volume() * self.block.volume()

    def warps_per_block(self) -> int:
        return ceil(self.block.volume() / WARP_SIZE)


@dataclass(frozen=True)
class StreamOp:
    name: str
    stream: int
    duration_us: float


def linear_thread_index(thread_idx: Dim3, block_dim: Dim3) -> int:
    """CUDA linearization: x is fastest, then y, then z."""
    return thread_idx.x + thread_idx.y * block_dim.x + thread_idx.z * block_dim.x * block_dim.y


def global_thread_id_1d(block_idx_x: int, thread_idx_x: int, block_dim_x: int) -> int:
    return block_idx_x * block_dim_x + thread_idx_x


def warp_and_lane(linear_tid: int) -> tuple[int, int]:
    return linear_tid // WARP_SIZE, linear_tid % WARP_SIZE


def launch_1d(problem_size: int, threads_per_block: int = 256, stream: int = 0) -> LaunchConfig:
    blocks = ceil(problem_size / threads_per_block)
    return LaunchConfig(grid=Dim3(blocks), block=Dim3(threads_per_block), stream=stream)


def schedule_blocks(n_blocks: int, n_sms: int, resident_blocks_per_sm: int) -> list[list[int]]:
    """Show waves of block scheduling without implying an architectural guarantee."""
    if n_sms <= 0 or resident_blocks_per_sm <= 0:
        raise ValueError("n_sms and resident_blocks_per_sm must be positive")
    capacity = n_sms * resident_blocks_per_sm
    return [list(range(start, min(start + capacity, n_blocks)))
            for start in range(0, n_blocks, capacity)]


def stream_timeline(ops: list[StreamOp]) -> list[dict]:
    """Operations in one stream are ordered; different streams may overlap."""
    stream_end: dict[int, float] = {}
    timeline = []
    for op in ops:
        start = stream_end.get(op.stream, 0.0)
        end = start + op.duration_us
        stream_end[op.stream] = end
        timeline.append({
            "op": op.name,
            "stream": op.stream,
            "start_us": start,
            "end_us": end,
        })
    return timeline


def graph_submission_overhead_us(
    n_ops: int,
    repeats: int,
    *,
    per_op_launch_us: float = 5.0,
    instantiate_us: float = 80.0,
    graph_launch_us: float = 6.0,
) -> dict:
    """Compare repeated host launches with one captured CUDA graph."""
    normal = repeats * n_ops * per_op_launch_us
    graphed = instantiate_us + repeats * graph_launch_us
    return {
        "normal_us": normal,
        "graph_us": graphed,
        "speedup": normal / graphed if graphed else float("inf"),
    }


def occupancy_from_threads(
    threads_per_block: int,
    *,
    max_threads_per_sm: int = 2048,
    max_blocks_per_sm: int = 32,
) -> dict:
    """Resource-light occupancy estimate using only thread and block limits."""
    blocks_by_threads = max_threads_per_sm // threads_per_block
    blocks = min(max_blocks_per_sm, blocks_by_threads)
    active_threads = blocks * threads_per_block
    return {
        "blocks_per_sm": blocks,
        "active_threads": active_threads,
        "occupancy": active_threads / max_threads_per_sm,
    }


def _self_test() -> None:
    cfg = launch_1d(10_000, 256)
    assert cfg.grid.x == 40
    assert cfg.total_threads() == 10_240
    assert cfg.warps_per_block() == 8

    assert global_thread_id_1d(3, 7, 256) == 775
    assert linear_thread_index(Dim3(3, 2, 1), Dim3(8, 4, 2)) == 3 + 2 * 8 + 1 * 8 * 4
    assert warp_and_lane(63) == (1, 31)
    assert warp_and_lane(64) == (2, 0)

    waves = schedule_blocks(n_blocks=20, n_sms=4, resident_blocks_per_sm=2)
    assert len(waves) == 3
    assert waves[0] == list(range(8))

    timeline = stream_timeline([
        StreamOp("h2d", 0, 10),
        StreamOp("kernel_a", 0, 20),
        StreamOp("kernel_b", 1, 15),
    ])
    assert timeline[1]["start_us"] == 10
    assert timeline[2]["start_us"] == 0

    graph = graph_submission_overhead_us(n_ops=10, repeats=100)
    assert graph["speedup"] > 5.0, graph

    occ_768 = occupancy_from_threads(768)
    assert round(occ_768["occupancy"], 2) == 0.75
    occ_32 = occupancy_from_threads(32)
    assert round(occ_32["occupancy"], 2) == 0.50
    print("[OK] cuda_original_minimal (launch, streams, graphs, occupancy)")


if __name__ == "__main__":
    _self_test()
