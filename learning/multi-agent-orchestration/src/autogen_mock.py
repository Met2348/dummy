"""AutoGen-style ConversableAgent + GroupChat mock."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ConversableAgent:
    name: str
    system_message: str
    reply_fn: Callable[[list[dict]], str]
    history: list[dict] = field(default_factory=list)

    def receive(self, message: dict) -> str:
        self.history.append(message)
        return self.reply_fn(self.history)


@dataclass
class GroupChat:
    agents: list[ConversableAgent]
    max_round: int = 10
    messages: list[dict] = field(default_factory=list)


class GroupChatManager:
    def __init__(self, chat: GroupChat, selector_fn: Callable[[list[dict], list[str]], str] | None = None):
        self.chat = chat
        self.selector = selector_fn or self._default_selector

    def _default_selector(self, history: list[dict], names: list[str]) -> str:
        return names[len(history) % len(names)]

    def run(self, initial_message: str) -> list[dict]:
        names = [a.name for a in self.chat.agents]
        self.chat.messages.append({"from": "user", "content": initial_message})

        for _ in range(self.chat.max_round):
            next_name = self.selector(self.chat.messages, names)
            if next_name == "TERMINATE":
                break
            agent = next(a for a in self.chat.agents if a.name == next_name)
            reply = agent.receive({"history": list(self.chat.messages)})
            self.chat.messages.append({"from": next_name, "content": reply})
            if "TERMINATE" in reply.upper():
                break
        return self.chat.messages


def _self_test() -> None:
    r_idx = [0]
    def r_reply(hist):
        r_idx[0] += 1
        return "research done" if r_idx[0] < 2 else "no more research"

    w_idx = [0]
    def w_reply(hist):
        w_idx[0] += 1
        return "written" if w_idx[0] < 2 else "DONE TERMINATE"

    researcher = ConversableAgent("researcher", "You research.", r_reply)
    writer = ConversableAgent("writer", "You write.", w_reply)
    chat = GroupChat(agents=[researcher, writer], max_round=5)
    mgr = GroupChatManager(chat)

    msgs = mgr.run("Write a blog about LLM")
    assert msgs[0]["content"] == "Write a blog about LLM"
    assert any(m["from"] == "researcher" for m in msgs)
    assert any(m["from"] == "writer" for m in msgs)
    assert "TERMINATE" in msgs[-1]["content"].upper()
    print("[OK] autogen_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
