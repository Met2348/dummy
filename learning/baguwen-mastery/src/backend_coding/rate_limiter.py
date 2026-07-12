"""令牌桶(token bucket)限流算法，纯 stdlib 实现。

时间推进用整数 tick 手动驱动（advance()），不依赖 time.sleep/time.time，
保证 _self_test() 的断言完全确定性、可复现。
"""
from __future__ import annotations


class TokenBucket:
    """容量为 capacity 的令牌桶，每推进 1 个 tick 恢复 refill_per_tick 个令牌。"""

    def __init__(self, capacity: int, refill_per_tick: int) -> None:
        self.capacity = capacity
        self.refill_per_tick = refill_per_tick
        self.tokens = capacity   # 初始桶是满的
        self.tick = 0

    def advance(self, ticks: int = 1) -> None:
        """推进 ticks 个整数时间步，按 refill_per_tick 恢复令牌，不超过容量。"""
        if ticks < 0:
            raise ValueError("ticks 不能为负")
        self.tick += ticks
        self.tokens = min(self.capacity, self.tokens + ticks * self.refill_per_tick)

    def allow(self, cost: int = 1) -> bool:
        """尝试消耗 cost 个令牌；桶内充足则放行并扣减，否则拒绝。"""
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def _self_test() -> None:
    bucket = TokenBucket(capacity=10, refill_per_tick=2)

    # 桶初始是满的(10个令牌)，中间不推进 tick：连续请求 15 次，
    # 前 10 次消耗完容量应通过，后 5 次桶已空应被拒绝。
    results = [bucket.allow() for _ in range(15)]
    assert results[:10] == [True] * 10, "前10次请求应在桶满(容量10)时全部通过"
    assert results[10:] == [False] * 5, "桶耗尽后应拒绝后续请求"
    assert bucket.tokens == 0

    # 推进 3 个 tick，每 tick 恢复 2 个令牌 -> 恢复 6 个(不超过容量10)。
    bucket.advance(3)
    assert bucket.tokens == 6
    assert bucket.tick == 3

    results_2 = [bucket.allow() for _ in range(8)]
    assert results_2[:6] == [True] * 6, "恢复的6个令牌应能通过6次请求"
    assert results_2[6:] == [False] * 2, "超出恢复量的请求应被拒绝"
    assert bucket.tokens == 0

    # 推进足够多 tick 应恢复到满容量，且不会因为恢复量超额而溢出。
    bucket.advance(100)
    assert bucket.tokens == bucket.capacity == 10
    assert bucket.tick == 103

    print(
        "[PASS] rate_limiter: token bucket 容量10/每tick恢复2 "
        "+ 15连发前10过后5拒 + tick推进恢复 + 满容量不溢出"
    )


if __name__ == "__main__":
    _self_test()
