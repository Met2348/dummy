"""rate_limiter_spec 的纯断言检验：不 import solutions/。tick 全部用确定性整数，不依赖真实时钟。"""
from __future__ import annotations


def check(target) -> None:
    limiter_cls = target  # target 就是 TokenBucketRateLimiter 这个类本身

    # ---- 1) 整数速率: capacity=5, refill_rate=1/tick ----
    lim = limiter_cls(capacity=5, refill_rate=1)

    # 桶初始是满的：tick=0 时连续 5 次突发请求应该全部放行，第 6 次应该被拒绝
    burst = [lim.allow(0) for _ in range(6)]
    assert burst == [True, True, True, True, True, False], (
        f"初始桶容量内的突发流量应该全放行，超过容量应该拒绝，实得 {burst}"
    )

    # 经过 3 个 tick，补充 3*1=3 个令牌（不超过容量），应该恰好能放行 3 次，第 4 次拒绝
    r2 = [lim.allow(3) for _ in range(4)]
    assert r2 == [True, True, True, False], (
        f"经过 3 tick 应该恰好补充 3 个令牌，实得 {r2}"
    )

    # 经过很长时间(100 tick)，补充应该封顶在 capacity=5，而不是无限增长
    r3 = [lim.allow(103) for _ in range(6)]
    assert r3 == [True, True, True, True, True, False], (
        f"长时间空闲后令牌补充应该封顶在 capacity，不能无限累积，实得 {r3}"
    )

    # ---- 2) 小数速率: capacity=2, refill_rate=0.5/tick，验证跨多个 tick 的分数累积 ----
    lim2 = limiter_cls(capacity=2, refill_rate=0.5)
    r4 = [lim2.allow(0) for _ in range(3)]
    assert r4 == [True, True, False], f"初始桶容量=2 应该恰好放行 2 次，实得 {r4}"

    # 只过 1 个 tick，补充 0.5 个令牌，不够 1 个，应该继续拒绝
    assert lim2.allow(1) is False, "只补充了 0.5 个令牌(<1)，这次请求应该被拒绝"

    # 再过 1 个 tick（累计 2 个 tick，补充 0.5+0.5=1.0 个令牌），这次应该放行
    assert lim2.allow(2) is True, "累计 2 个 tick 应该补充满 1 个令牌，这次请求应该放行"

    # ---- 3) 被拒绝的请求也必须正确结算 tick，不能让下次补充重复计算这段时间 ----
    lim3 = limiter_cls(capacity=1, refill_rate=1)
    assert lim3.allow(0) is True         # 用掉唯一的 1 个令牌
    assert lim3.allow(0) is False        # 同一 tick 内，没有补充，应该拒绝
    assert lim3.allow(0) is False        # 再次拒绝，不应该产生任何"偷跑"的令牌
    assert lim3.allow(1) is True         # 过了 1 个 tick，补满 1 个令牌，这次放行
    assert lim3.allow(1) is False        # 又立刻用完，同 tick 内应该拒绝
