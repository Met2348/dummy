"""WebArena (CMU 2024) — mock with 1 fake shop browser.

Real bench: shopping/gitlab/reddit/CMS-style real browser tasks, ~800
episodes. Agent observes DOM, emits actions (click/type/select).

Our mock: static HTML state, action verbs (browse/add/checkout). Tests
that agent reaches a specific goal state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from common import ModelFn, make_mock_model


@dataclass
class ShopState:
    page: str = "home"     # home / item / cart / checkout
    cart_items: List[str] = field(default_factory=list)
    confirmed: bool = False


@dataclass
class WebTask:
    qid: str
    goal: str
    expected: Dict   # required end-state e.g. {"confirmed": True, "cart": ["book"]}


_TASKS: List[WebTask] = [
    WebTask(
        qid="web_1",
        goal="Add the book to cart and check out.",
        expected={"confirmed": True, "cart_items": ["book"]},
    ),
]


def step(state: ShopState, action: str) -> ShopState:
    a = action.strip().lower()
    if a == "go to item":
        state.page = "item"
    elif a == "add to cart":
        if "book" not in state.cart_items:
            state.cart_items.append("book")
        state.page = "cart"
    elif a == "go to checkout":
        state.page = "checkout"
    elif a == "confirm order":
        if state.cart_items:
            state.confirmed = True
    return state


def parse_actions(text: str) -> List[str]:
    """Each non-empty line is an action."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def goal_reached(state: ShopState, expected: Dict) -> bool:
    if expected.get("confirmed") != state.confirmed:
        return False
    if expected.get("cart_items") != state.cart_items:
        return False
    return True


def run_webarena_mock(model: ModelFn) -> List[Dict]:
    rs = []
    for t in _TASKS:
        prompt = (f"[qid={t.qid}]\nGoal: {t.goal}\n"
                  "Output one action per line. Allowed actions: "
                  "'go to item', 'add to cart', 'go to checkout', 'confirm order'.")
        text = model(prompt, 256)
        state = ShopState()
        for a in parse_actions(text):
            state = step(state, a)
        rs.append({"qid": t.qid, "actions": parse_actions(text),
                    "passed": goal_reached(state, t.expected)})
    return rs


def _self_test() -> int:
    # Empty model fails
    rs = run_webarena_mock(make_mock_model({}, default=""))
    assert not rs[0]["passed"]
    # Correct script passes
    m = make_mock_model({"web_1":
        "go to item\nadd to cart\ngo to checkout\nconfirm order"})
    rs2 = run_webarena_mock(m)
    assert rs2[0]["passed"], rs2[0]
    return 0


def _demo() -> None:
    """Visible demo: drive the real shop state machine to the goal state."""
    print(f"WebArena mock: {len(_TASKS)} shopping task — actions drive a real ShopState machine")
    empty = run_webarena_mock(make_mock_model({}, default=""))
    good_script = "go to item\nadd to cart\ngo to checkout\nconfirm order"
    good = run_webarena_mock(make_mock_model({"web_1": good_script}))
    # Show the end-state reached by replaying the good script.
    st = ShopState()
    for a in parse_actions(good_script):
        st = step(st, a)
    print(f"  no-action     -> passed={empty[0]['passed']}")
    print(f"  4-step script -> passed={good[0]['passed']}  "
          f"end-state: page={st.page}, cart={st.cart_items}, confirmed={st.confirmed}")
    print("  -> goal reached iff replayed actions yield the required end-state (no hardcoded score).")


if __name__ == "__main__":
    f = _self_test()
    print(f"webarena_mock.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
