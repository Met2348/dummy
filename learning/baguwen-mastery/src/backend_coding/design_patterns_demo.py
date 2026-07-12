"""设计模式代码验证模块：手写 5 个模式的最小可跑实现，配套 `_self_test()`
断言各自的行为契约（对应 `backend_qa/qbank_design_patterns.py` 里的问答）。

覆盖：
- 单例（Singleton）：线程安全版本，用 threading.Lock 做双重检查锁。
- 工厂（Factory Method）：根据参数创建不同的具体产品子类。
- 观察者（Observer）：notify() 广播给所有已订阅的观察者。
- 装饰器（Decorator）：多层装饰叠加，顺序影响结果。
- 策略（Strategy）：同一上下文接口，切换策略产生不同结果。
"""
from __future__ import annotations

import math
import threading


# ---------------------------------------------------------------------------
# 1. 单例模式：双重检查锁 + threading.Lock（对应 be-dp-01~04）
# ---------------------------------------------------------------------------
class Singleton:
    """线程安全的懒加载单例：__new__ 里做双重检查锁。"""

    _instance: "Singleton | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "Singleton":
        if cls._instance is None:                # 第一次判空：避免每次调用都抢锁
            with cls._lock:
                if cls._instance is None:         # 第二次判空：防止排队线程重复创建
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 真实场景里这里可能有较重的初始化逻辑；__new__ 已经保证只会真正构造一次，
        # 但 __init__ 每次 Singleton() 调用都会执行，这里保持幂等（无副作用）即可。
        pass


# ---------------------------------------------------------------------------
# 2. 工厂模式：Factory Method（对应 be-dp-05~07）
# ---------------------------------------------------------------------------
class Shape:
    def area(self) -> float:
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2


class Square(Shape):
    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return self.side ** 2


class ShapeFactory:
    """简化版工厂：按 kind 分发到具体产品子类。"""

    _REGISTRY: dict[str, type[Shape]] = {"circle": Circle, "square": Square}

    @classmethod
    def create(cls, kind: str, *args: float) -> Shape:
        if kind not in cls._REGISTRY:
            raise ValueError(f"未知的形状类型: {kind}")
        return cls._REGISTRY[kind](*args)


# ---------------------------------------------------------------------------
# 3. 观察者模式（对应 be-dp-08）
# ---------------------------------------------------------------------------
class Observer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.received: list[str] = []

    def update(self, event: str) -> None:
        self.received.append(event)


class Subject:
    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    def notify(self, event: str) -> None:
        for observer in self._observers:
            observer.update(event)


# ---------------------------------------------------------------------------
# 4. 装饰器模式（对应 be-dp-09）
# ---------------------------------------------------------------------------
class Coffee:
    def cost(self) -> float:
        return 10.0

    def desc(self) -> str:
        return "Coffee"


class _CoffeeDecorator(Coffee):
    def __init__(self, inner: Coffee) -> None:
        self._inner = inner


class MilkDecorator(_CoffeeDecorator):
    def cost(self) -> float:
        return self._inner.cost() + 2.0

    def desc(self) -> str:
        return self._inner.desc() + "+Milk"


class SugarDecorator(_CoffeeDecorator):
    def cost(self) -> float:
        return self._inner.cost() + 1.0

    def desc(self) -> str:
        return self._inner.desc() + "+Sugar"


# ---------------------------------------------------------------------------
# 5. 策略模式（对应 be-dp-10）
# ---------------------------------------------------------------------------
class DiscountStrategy:
    def apply(self, price: float) -> float:
        raise NotImplementedError


class NoDiscount(DiscountStrategy):
    def apply(self, price: float) -> float:
        return price


class PercentOff(DiscountStrategy):
    def __init__(self, pct: float) -> None:
        self.pct = pct

    def apply(self, price: float) -> float:
        return price * (1 - self.pct)


class Order:
    """上下文：持有策略接口引用，切换策略不改自身代码。"""

    def __init__(self, strategy: DiscountStrategy) -> None:
        self.strategy = strategy

    def final_price(self, price: float) -> float:
        return self.strategy.apply(price)


def _self_test() -> None:
    # --- 1. 单例：多线程并发调用应始终拿到同一个对象 ---
    Singleton._instance = None  # 保证测试独立于模块导入顺序
    results: list[Singleton] = []
    results_lock = threading.Lock()

    def _grab() -> None:
        inst = Singleton()
        with results_lock:
            results.append(inst)

    threads = [threading.Thread(target=_grab) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 16
    first = results[0]
    assert all(r is first for r in results), "并发下 Singleton() 返回了不同实例"
    assert Singleton() is first

    # --- 2. 工厂：不同参数应创建出正确的不同子类实例 ---
    circle = ShapeFactory.create("circle", 2.0)
    square = ShapeFactory.create("square", 3.0)
    assert isinstance(circle, Circle) and not isinstance(circle, Square)
    assert isinstance(square, Square) and not isinstance(square, Circle)
    assert abs(circle.area() - math.pi * 4.0) < 1e-9
    assert square.area() == 9.0
    try:
        ShapeFactory.create("triangle")
        raise AssertionError("未知类型应抛出 ValueError")
    except ValueError:
        pass

    # --- 3. 观察者：notify() 后所有订阅者都应收到正确的事件数据 ---
    subject = Subject()
    obs_a, obs_b, obs_c = Observer("A"), Observer("B"), Observer("C")
    for obs in (obs_a, obs_b, obs_c):
        subject.subscribe(obs)
    subject.notify("event-1")
    subject.notify("event-2")
    for obs in (obs_a, obs_b, obs_c):
        assert obs.received == ["event-1", "event-2"], obs.received

    # 未订阅的观察者不应该收到通知
    late_obs = Observer("late")
    assert late_obs.received == []

    # --- 4. 装饰器：多层装饰叠加，且顺序影响 desc 但不影响总价（cost 满足交换律） ---
    order1 = MilkDecorator(SugarDecorator(Coffee()))   # Coffee -> +Sugar -> +Milk
    order2 = SugarDecorator(MilkDecorator(Coffee()))   # Coffee -> +Milk -> +Sugar
    assert order1.cost() == order2.cost() == 13.0
    assert order1.desc() == "Coffee+Sugar+Milk"
    assert order2.desc() == "Coffee+Milk+Sugar"
    assert order1.desc() != order2.desc(), "叠加顺序不同，desc 应该不同"

    # --- 5. 策略：同一个 Order 上下文，切换策略应产生不同结果 ---
    order = Order(NoDiscount())
    assert order.final_price(100.0) == 100.0
    order.strategy = PercentOff(0.2)
    assert order.final_price(100.0) == 80.0
    order.strategy = PercentOff(0.5)
    assert order.final_price(100.0) == 50.0

    print(
        "[PASS] design_patterns_demo: 单例(并发16线程同一实例) + 工厂(2类正确分发) "
        "+ 观察者(3订阅者收齐2事件) + 装饰器(叠加顺序影响结果) + 策略(3种策略结果各异)"
    )


if __name__ == "__main__":
    _self_test()
