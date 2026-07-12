"""rate_limiter_spec 的参考实现。懒惰刷新（lazy refill），无后台线程/真实时钟依赖。"""
from __future__ import annotations


class TokenBucketRateLimiter:
    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)   # 桶初始状态：满
        self._last_tick = 0

    def allow(self, tick: int) -> bool:
        elapsed = tick - self._last_tick
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_tick = tick
        # 不管放不放行，last_tick 都已经在上面更新过了（elapsed<=0 时本来就不需要动）

        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False
