"""Triton-style kernel — block-pointer + autotune."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TritonConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int = 4
    num_stages: int = 3      # software pipelining depth

    def smem_bytes(self, dtype_bytes: int = 2) -> int:
        """Total SMEM for A and B tiles, double-buffered by num_stages."""
        return dtype_bytes * self.num_stages * (
            self.block_m * self.block_k + self.block_k * self.block_n
        )


CONFIGS = [
    TritonConfig(64,  64,  32, num_warps=4, num_stages=3),
    TritonConfig(128, 64,  32, num_warps=4, num_stages=3),
    TritonConfig(128, 128, 32, num_warps=8, num_stages=3),
    TritonConfig(128, 256, 32, num_warps=8, num_stages=4),
    TritonConfig(256, 128, 32, num_warps=8, num_stages=4),
]


def autotune(M: int, N: int, K: int, smem_limit_kb: int = 228,
             configs: list[TritonConfig] = None) -> TritonConfig:
    """Pick config with best estimated throughput within SMEM budget."""
    cfgs = configs or CONFIGS
    valid = [c for c in cfgs if c.smem_bytes() <= smem_limit_kb * 1024]
    if not valid:
        raise ValueError(f"no valid config for SMEM {smem_limit_kb} KB")

    def score(c: TritonConfig) -> float:
        # Reuse heuristic: larger tile = more reuse, more parallelism
        reuse = c.block_m * c.block_n / max(1, c.block_m + c.block_n)
        n_tiles = ((M + c.block_m - 1) // c.block_m) * ((N + c.block_n - 1) // c.block_n)
        # Want enough tiles to keep all 132 SMs busy
        parallelism_penalty = 0 if n_tiles >= 132 else (132 - n_tiles) * 0.1
        return reuse * c.num_stages - parallelism_penalty

    return max(valid, key=score)


def _self_test() -> None:
    # Big GEMM 4k³ → prefer big tile
    big = autotune(4096, 4096, 4096)
    assert big.block_m * big.block_n >= 128 * 128, big

    # Small M (256) → tall-thin config
    small = autotune(256, 4096, 4096)
    # SMEM constraint
    huge = TritonConfig(512, 512, 64, num_stages=4)
    assert huge.smem_bytes() > 228 * 1024, huge.smem_bytes()
    try:
        autotune(4096, 4096, 4096, smem_limit_kb=1, configs=[huge])
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print(f"[OK] triton_style (big GEMM picked {big.block_m}x{big.block_n})")


if __name__ == "__main__":
    _self_test()
