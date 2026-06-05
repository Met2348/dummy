"""Pub-sub message bus."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class BusMessage:
    topic: str
    from_agent: str
    payload: Any


class MessageBus:
    def __init__(self):
        self.subs: dict[str, list[tuple[str, Callable]]] = {}
        self.history: list[BusMessage] = []

    def subscribe(self, topic: str, agent_name: str, callback: Callable[[BusMessage], None]) -> None:
        self.subs.setdefault(topic, []).append((agent_name, callback))

    def publish(self, topic: str, payload: Any, from_agent: str = "?") -> int:
        msg = BusMessage(topic=topic, from_agent=from_agent, payload=payload)
        self.history.append(msg)
        delivered = 0
        for name, cb in self.subs.get(topic, []):
            if name == from_agent:
                continue
            cb(msg)
            delivered += 1
        return delivered


def _self_test() -> None:
    bus = MessageBus()
    received: list[BusMessage] = []
    bus.subscribe("search_done", "writer", lambda m: received.append(m))
    bus.subscribe("search_done", "critic", lambda m: received.append(m))

    n = bus.publish("search_done", {"results": ["a", "b"]}, from_agent="researcher")
    assert n == 2
    assert len(received) == 2
    assert received[0].topic == "search_done"
    assert received[0].payload["results"] == ["a", "b"]

    bus.publish("search_done", "second", from_agent="researcher")
    assert len(received) == 4

    bus.publish("noone_listening", {})
    assert len(received) == 4

    assert len(bus.history) == 3
    print("[OK] message_bus._self_test passed")


if __name__ == "__main__":
    _self_test()
