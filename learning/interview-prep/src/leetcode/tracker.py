"""间隔复习追踪器（SM-2 简版）。确定性：以整数"天"驱动，无 Date.now/随机。

用法：add 一道题 → 每次做完 review(id, quality 0-5, today) → due(today) 列出今天该复习的。
答得好间隔拉长、答得差打回重来——把 100 道题的复习节奏交给算法，而非蛮背。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Card:
    id: str
    pattern: str
    name: str
    ease: float = 2.5
    interval: int = 0        # 天
    reps: int = 0
    due_day: int = 0


class ReviewTracker:
    def __init__(self):
        self.cards: dict[str, Card] = {}

    def add(self, id: str, pattern: str, name: str, today: int = 0) -> None:
        if id not in self.cards:
            self.cards[id] = Card(id=id, pattern=pattern, name=name, due_day=today)

    def review(self, id: str, quality: int, today: int) -> Card:
        """quality: 0-5（<3 视为没做出来，打回）。返回更新后的 card。"""
        c = self.cards[id]
        if quality < 3:                          # 失败：间隔归 1，重头来
            c.reps = 0
            c.interval = 1
        else:                                    # 成功：间隔按 SM-2 递增
            c.reps += 1
            if c.reps == 1:
                c.interval = 1
            elif c.reps == 2:
                c.interval = 6
            else:
                c.interval = round(c.interval * c.ease)
        # ease 因子调整（答得越好越大，下限 1.3）
        c.ease = max(1.3, c.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        c.due_day = today + c.interval
        return c

    def due(self, today: int) -> list[str]:
        """今天（含之前）该复习的题 id，按 due_day 升序。"""
        return [c.id for c in sorted(self.cards.values(), key=lambda x: x.due_day)
                if c.due_day <= today]

    def to_json(self) -> str:
        return json.dumps({k: asdict(v) for k, v in self.cards.items()}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ReviewTracker":
        t = cls()
        for k, v in json.loads(s).items():
            t.cards[k] = Card(**v)
        return t


def _self_test() -> None:
    t = ReviewTracker()
    t.add("lc3", "滑动窗口", "无重复最长子串")
    t.add("lc1", "哈希频次", "两数之和")

    # 全部初始 due 于第 0 天
    assert set(t.due(0)) == {"lc3", "lc1"}

    # lc3 答得好：间隔拉长，第 0 天不再 due，未来才 due
    t.review("lc3", quality=5, today=0)
    assert "lc3" not in t.due(0)
    assert "lc3" in t.due(10)               # 未来会回来

    # lc1 答得差：打回，间隔=1，第 1 天又 due
    t.review("lc1", quality=1, today=0)
    assert t.cards["lc1"].interval == 1
    assert "lc1" in t.due(1)

    # 连续答好 ease 上升、间隔加速
    t.review("lc3", 5, 1); t.review("lc3", 5, 2)
    assert t.cards["lc3"].reps == 3
    assert t.cards["lc3"].interval >= 6

    # 序列化往返
    t2 = ReviewTracker.from_json(t.to_json())
    assert t2.cards["lc3"].reps == t.cards["lc3"].reps
    print("[PASS] tracker: 初始due + 成功拉长 + 失败打回 + 序列化往返")


if __name__ == "__main__":
    _self_test()
