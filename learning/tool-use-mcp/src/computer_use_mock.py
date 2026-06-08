"""Mock Computer Use - screenshot + click/type actions."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MockComputerState:
    width: int = 1024
    height: int = 768
    cursor: tuple[int, int] = (0, 0)
    screen_text: str = ""
    action_log: list[tuple] = field(default_factory=list)


class MockComputer:
    VALID_ACTIONS = {"screenshot", "left_click", "type", "key", "cursor_position"}

    def __init__(self, width: int = 1024, height: int = 768):
        self.state = MockComputerState(width=width, height=height)

    def execute(self, action: dict) -> dict:
        a_type = action.get("type")
        if a_type not in self.VALID_ACTIONS:
            return {"ok": False, "error": f"invalid action: {a_type}"}

        if a_type == "screenshot":
            return {
                "ok": True,
                "image": f"<screenshot {self.state.width}x{self.state.height}: '{self.state.screen_text[:40]}'>",
                "cursor": self.state.cursor,
            }
        if a_type == "left_click":
            coord = tuple(action.get("coordinate", [0, 0]))
            if not (0 <= coord[0] < self.state.width and 0 <= coord[1] < self.state.height):
                return {"ok": False, "error": "coordinate out of bounds"}
            self.state.cursor = coord
            self.state.action_log.append(("click", coord))
            return {"ok": True, "cursor": coord}
        if a_type == "type":
            text = action.get("text", "")
            self.state.screen_text += text
            self.state.action_log.append(("type", text))
            return {"ok": True, "typed": text}
        if a_type == "key":
            key = action.get("key", "")
            self.state.action_log.append(("key", key))
            if key.lower() == "return":
                self.state.screen_text += "\n"
            return {"ok": True, "key": key}
        if a_type == "cursor_position":
            return {"ok": True, "cursor": self.state.cursor}
        return {"ok": False, "error": "unhandled"}


def _self_test() -> None:
    pc = MockComputer()
    r = pc.execute({"type": "screenshot"})
    assert r["ok"] and "screenshot" in r["image"]

    r = pc.execute({"type": "left_click", "coordinate": [100, 200]})
    assert r["ok"] and r["cursor"] == (100, 200)

    r = pc.execute({"type": "type", "text": "hello"})
    assert r["ok"]
    assert "hello" in pc.state.screen_text

    r = pc.execute({"type": "left_click", "coordinate": [-1, 5]})
    assert not r["ok"] and "out of bounds" in r["error"]

    r = pc.execute({"type": "invalid_action"})
    assert not r["ok"]

    assert len(pc.state.action_log) >= 2
    print("[OK] computer_use_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
