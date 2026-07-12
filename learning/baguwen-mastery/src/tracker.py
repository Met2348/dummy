"""间隔复习追踪器（SM-2 简版）。确定性：以整数"天"驱动，无 Date.now/随机。

与 leetcode-mastery/src/tracker.py 算法完全一致，这里的 seed_from_qa() 面向
QA 对象（八股问答卡片）而非 LeetCode Problem。用法：seed 完 -> 每次自测完一题
review(id, quality 0-5, today) -> due(today) 列出今天该复习的。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Card:
    id: str
    category: str
    name: str
    ease: float = 2.5
    interval: int = 0        # 天
    reps: int = 0
    due_day: int = 0


class ReviewTracker:
    def __init__(self):
        self.cards: dict[str, Card] = {}

    def add(self, id: str, category: str, name: str, today: int = 0) -> None:
        if id not in self.cards:
            self.cards[id] = Card(id=id, category=category, name=name, due_day=today)

    def review(self, id: str, quality: int, today: int) -> Card:
        """quality: 0-5（<3 视为没答上来，打回）。返回更新后的 card。"""
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


def seed_from_qa(tracker: ReviewTracker, qa_list) -> None:
    """把 QA 列表（ai_qa.ALL_QA + backend_qa.ALL_QA）一次性 add 进 tracker，全部初始 due 于第 0 天。"""
    for qa in qa_list:
        tracker.add(qa.id, qa.cat, qa.q, today=0)


def _self_test() -> None:
    import os
    import sys
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SRC_DIR)
    from ai_qa import ALL_QA as AI_QA
    from backend_qa import ALL_QA as BE_QA
    all_qa = AI_QA + BE_QA
    n = len(all_qa)

    t = ReviewTracker()
    seed_from_qa(t, all_qa)
    assert len(t.cards) == n
    assert len(t.due(0)) == n          # 全部初始 due 于第 0 天

    # 答得好：间隔拉长，第 0 天不再 due，未来才 due
    good_id = all_qa[0].id
    t.review(good_id, quality=5, today=0)
    assert good_id not in t.due(0)
    assert good_id in t.due(10)

    # 答得差：打回，间隔=1，第 1 天又 due
    bad_id = all_qa[1].id
    t.review(bad_id, quality=1, today=0)
    assert t.cards[bad_id].interval == 1
    assert bad_id in t.due(1)

    # 连续答好 ease 上升、间隔加速
    t.review(good_id, 5, 1)
    t.review(good_id, 5, 2)
    assert t.cards[good_id].reps == 3
    assert t.cards[good_id].interval >= 6

    # 序列化往返
    t2 = ReviewTracker.from_json(t.to_json())
    assert t2.cards[good_id].reps == t.cards[good_id].reps
    assert len(t2.cards) == n
    print(f"[PASS] tracker: {n}题种子 + 成功拉长 + 失败打回 + 序列化往返")


if __name__ == "__main__":
    _self_test()
