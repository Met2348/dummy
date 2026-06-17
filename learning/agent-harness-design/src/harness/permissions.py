"""Permission system — the gate between an agent's intent and real side effects.

Modes:
  * auto      — allow everything except the deny-list
  * readonly  — allow only read-only tools (or allow-list); block writes
  * ask       — route through an ask_handler, with allow/deny-list shortcuts
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    action: str  # "allow" | "deny"
    reason: str = ""


class PermissionManager:
    def __init__(self, mode: str = "auto", allow=None, deny=None, ask_handler=None):
        self.mode = mode
        self.allow = set(allow or [])
        self.deny = set(deny or [])
        self.ask_handler = ask_handler or (lambda name, args: True)
        self.log: list = []

    def check(self, tool, args: dict | None = None) -> Decision:
        name = getattr(tool, "name", tool)
        read_only = getattr(tool, "read_only", False)
        d = self._decide(name, read_only, args)
        self.log.append((name, d.action, d.reason))
        return d

    def _decide(self, name, read_only, args) -> Decision:
        if name in self.deny:
            return Decision("deny", "deny-list")
        if name in self.allow:
            return Decision("allow", "allow-list")
        if self.mode == "auto":
            return Decision("allow", "auto")
        if self.mode == "readonly":
            return Decision("allow", "read-only") if read_only \
                else Decision("deny", "writes blocked in readonly mode")
        if self.mode == "ask":
            return Decision("allow", "approved") if self.ask_handler(name, args) \
                else Decision("deny", "user declined")
        return Decision("deny", f"unknown mode: {self.mode}")


def _self_test() -> None:
    from .tools import Tool

    write = Tool("write_file", "writes", lambda: None, read_only=False)
    read = Tool("read_file", "reads", lambda: None, read_only=True)

    auto = PermissionManager(mode="auto", deny=["rm_rf"])
    assert auto.check(write).action == "allow"
    assert auto.check("rm_rf").action == "deny"

    ro = PermissionManager(mode="readonly")
    assert ro.check(read).action == "allow"
    assert ro.check(write).action == "deny" and "readonly" in ro.check(write).reason

    asked = {"n": 0}

    def handler(name, args):
        asked["n"] += 1
        return name != "write_file"  # approve everything except write

    ask = PermissionManager(mode="ask", allow=["read_file"], ask_handler=handler)
    assert ask.check(read).action == "allow"          # allow-list shortcut, no prompt
    assert ask.check(write).action == "deny"          # handler declines
    assert asked["n"] == 1                              # only write hit the handler
    assert len(ask.log) == 2
    print("[OK] harness.permissions._self_test passed")


if __name__ == "__main__":
    _self_test()
