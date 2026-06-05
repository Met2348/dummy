"""User profile personalization."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    preferences: dict[str, str] = field(default_factory=dict)
    knowledge_level: dict[str, str] = field(default_factory=dict)
    style: dict[str, str] = field(default_factory=dict)
    domain: dict[str, str] = field(default_factory=dict)
    feedback_count: dict[str, int] = field(default_factory=dict)

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def set_knowledge(self, topic: str, level: str) -> None:
        if level not in {"none", "beginner", "intermediate", "expert"}:
            raise ValueError(f"invalid level: {level}")
        self.knowledge_level[topic] = level

    def set_style(self, key: str, value: str) -> None:
        self.style[key] = value

    def update_from_feedback(self, action: str, response_style: str) -> None:
        key = f"{action}:{response_style}"
        self.feedback_count[key] = self.feedback_count.get(key, 0) + 1
        if action == "thumbs_up" and self.feedback_count[key] >= 2:
            self.style.setdefault("validated", []).append(response_style)
        elif action == "thumbs_down" and self.feedback_count[key] >= 1:
            self.style["avoid_" + response_style] = "yes"

    def to_system_prompt(self) -> str:
        parts = ["USER PROFILE:"]
        if self.name:
            parts.append(f"- Name: {self.name}")
        for k, v in self.preferences.items():
            parts.append(f"- Preference [{k}]: {v}")
        for k, v in self.knowledge_level.items():
            parts.append(f"- Knowledge [{k}]: {v}")
        for k, v in self.style.items():
            parts.append(f"- Style [{k}]: {v}")
        for k, v in self.domain.items():
            parts.append(f"- Domain [{k}]: {v}")
        if len(parts) == 1:
            parts.append("- (no profile yet)")
        return "\n".join(parts)


def _self_test() -> None:
    p = UserProfile(user_id="alice")
    p.name = "Alice"
    p.set_preference("llm", "Anthropic Claude")
    p.set_knowledge("transformer", "expert")
    p.set_knowledge("rl", "beginner")
    p.set_style("language", "chinese")
    p.set_style("verbosity", "concise")

    prompt = p.to_system_prompt()
    assert "Alice" in prompt
    assert "Anthropic" in prompt
    assert "chinese" in prompt

    try:
        p.set_knowledge("rag", "guru")
        assert False, "should have raised"
    except ValueError:
        pass

    p.update_from_feedback("thumbs_up", "code_examples")
    p.update_from_feedback("thumbs_up", "code_examples")
    assert "validated" in p.style
    assert "code_examples" in p.style["validated"]

    p.update_from_feedback("thumbs_down", "long_intro")
    assert p.style.get("avoid_long_intro") == "yes"
    print("[OK] personalization._self_test passed")


if __name__ == "__main__":
    _self_test()
