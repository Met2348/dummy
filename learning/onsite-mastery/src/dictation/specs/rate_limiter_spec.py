"""Token Bucket 限流器，闭卷从零手写。

面试高频度 ****。系统设计白板常客，经常紧跟着问"token bucket 和 leaky bucket
的区别"、"怎么在分布式场景下做（Redis+Lua）"。这里只要求单机、单一时间线的版本，
用**离散 tick**（整数时间步）代替真实时钟，不需要 `time.sleep`/后台线程。

接口约定
--------
    class TokenBucketRateLimiter:
        def __init__(self, capacity: float, refill_rate: float): ...
        def allow(self, tick: int) -> bool: ...

    capacity    : 桶的最大容量（能攒的最多令牌数）
    refill_rate : 每个 tick 补充多少令牌（可以是小数，比如 0.5 个/tick）
    tick        : 调用时的"当前时间步"，是一个不减的整数（由调用方传入，不用
                  `time.time()`，方便做确定性单元测试）

    约定：**桶在 t=0 时刻是满的**（初始 tokens = capacity），这是大多数教科书
    实现的默认假设——一开始就允许一次到 capacity 大小的突发流量。

`allow(tick)` 语义
------------------
    1. 先按经过的 tick 数补充令牌：tokens = min(capacity, tokens + elapsed * refill_rate)，
       elapsed = tick - 上次结算的 tick。
    2. 如果 tokens >= 1，允许这次请求，消耗 1 个令牌，返回 True。
    3. 否则拒绝，返回 False。
    4. **不管这次请求是否被允许，"上次结算的 tick"都要更新**——否则被拒绝的
       请求会导致下一次补充时把这段时间重复计算一遍，人为放大了实际的补充速率。

面试常问
--------
- 为什么叫"token bucket"（而不是真的开一个定时器每隔固定时间加令牌）？
  —— 懒惰刷新（lazy refill）：不需要后台线程/定时器，只在每次请求到来时
  按经过的时间"补算"应该攒了多少令牌，省资源、也没有定时器抖动的问题。
- Token bucket 和 leaky bucket 的区别？—— token bucket 允许**突发**流量
  （只要桶里有攒够的令牌，可以一瞬间消耗完，允许短时间超过平均速率）；
  leaky bucket 是恒定速率流出，无论输入多突发，输出永远被拉平成常数速率，
  不允许突发。

常见实现陷阱
------------
1. **忘记 `min(capacity, ...)` 封顶**：长时间没有请求之后，tokens 会无限增长，
   下次来了一堆突发请求也全部放行，完全失去"限流"的意义——这是本题最容易漏掉、
   也最容易被面试官问到的一条。
2. **判断条件用 `> 0` 而不是 `>= 1`**：允许消耗一个"不完整"的令牌（比如只有
   0.3 个令牌也放行），这不符合"一个请求消耗一个完整令牌"的计数语义，除非
   面试官明确要支持分数请求。
3. **只在放行时才更新"上次结算 tick"**：被拒绝的请求如果不更新 tick，下次
   请求来的时候 elapsed 会被重复计算，相当于免费多送了一段时间的令牌。
4. **懒惰刷新 vs 真实定时器搞混**：不要在这道题里引入 `threading.Timer`/
   `time.sleep` 之类的真实时钟依赖——tick 是调用方显式传入的逻辑时间，
   这也是为了让测试完全确定性、可复现。
"""
from __future__ import annotations


class TokenBucketRateLimiter:
    """Token Bucket 限流器：懒惰刷新，tick 是调用方传入的离散整数时间步。"""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        raise NotImplementedError("闭卷手写：删除这行 raise，初始化桶容量/补充速率/初始令牌数/上次结算tick")

    def allow(self, tick: int) -> bool:
        """按经过的 tick 补充令牌后，判断这次请求能否放行。"""
        raise NotImplementedError("闭卷手写：删除这行 raise，实现 allow")
