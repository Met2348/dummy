"""Mem0 mock - extract / update / retrieve."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import hash_embed, cosine, tokenize


PREFERENCE_PATTERNS = [
    (r"i (prefer|like) ([\w\s]+?) (?:over|instead of|to) ([\w\s]+?)(?:[\.\,\!\?]|$)", "preference"),
    (r"i (prefer|like) ([\w\s]+?)(?:[\.\,\!\?]|$)", "preference"),
    (r"my name is (\w+)", "name"),
    (r"i('m| am) a ([\w\s]+?)(?:[\.\,\!\?]|$)", "role"),
    (r"i work on ([\w\s]+?)(?:[\.\,\!\?]|$)", "project"),
    (r"i('m| am) (using|on) ([\w\s]+?)(?:[\.\,\!\?]|$)", "tool_used"),
    (r"i switched (?:from [\w\s]+ )?to ([\w\s]+?)(?:[\.\,\!\?]|$)", "tool_used"),
]


def extract_facts(content: str) -> list[dict]:
    import re
    facts = []
    text = content.lower()
    seen_kinds: set[str] = set()
    for pat, kind in PREFERENCE_PATTERNS:
        if kind in seen_kinds:
            continue
        m = re.search(pat, text)
        if not m:
            continue
        groups = m.groups()
        if kind == "preference" and len(groups) >= 3:
            facts.append({"kind": "preference", "value": groups[1].strip(), "old": groups[2].strip()})
        elif kind == "preference":
            facts.append({"kind": "preference", "value": groups[1].strip()})
        elif kind == "name":
            facts.append({"kind": "name", "value": groups[0].strip().capitalize()})
        elif kind == "role":
            facts.append({"kind": "role", "value": groups[1].strip()})
        elif kind == "project":
            facts.append({"kind": "project", "value": groups[0].strip()})
        elif kind == "tool_used":
            value = groups[-1].strip()
            facts.append({"kind": "tool_used", "value": value})
        seen_kinds.add(kind)
    return facts


def decide_action(new_fact: dict, existing: list[dict]) -> str:
    for ex in existing:
        if ex["kind"] != new_fact["kind"]:
            continue
        if ex["value"].lower() == new_fact["value"].lower():
            return "NONE"
        return "UPDATE"
    return "ADD"


@dataclass
class Mem0Store:
    facts_by_user: dict[str, list[dict]] = field(default_factory=dict)

    def add(self, content: str, user_id: str) -> dict:
        existing = self.facts_by_user.setdefault(user_id, [])
        new_facts = extract_facts(content)
        actions = []
        for nf in new_facts:
            action = decide_action(nf, existing)
            actions.append((nf, action))
            if action == "ADD":
                nf["embedding"] = hash_embed(f"{nf['kind']} {nf['value']}")
                existing.append(nf)
            elif action == "UPDATE":
                for ex in existing:
                    if ex["kind"] == nf["kind"]:
                        ex["value"] = nf["value"]
                        ex["embedding"] = hash_embed(f"{nf['kind']} {nf['value']}")
                        break
        return {"actions": actions, "facts_after": list(existing)}

    def search(self, query: str, user_id: str, k: int = 3) -> list[dict]:
        q_vec = hash_embed(query)
        existing = self.facts_by_user.get(user_id, [])
        scored = [
            (f, cosine(q_vec, f.get("embedding", hash_embed(f["value"]))))
            for f in existing
        ]
        return [f for f, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]

    def get_preference(self, user_id: str, kind: str) -> str | None:
        for f in self.facts_by_user.get(user_id, []):
            if f["kind"] == kind:
                return f["value"]
        return None


def _self_test() -> None:
    store = Mem0Store()
    r1 = store.add("My name is Alice. I prefer Anthropic Claude over GPT.", "alice")
    actions = [a for _, a in r1["actions"]]
    assert "ADD" in actions

    r2 = store.add("I switched to Gemini now.", "alice")
    found_update = any(a == "UPDATE" for _, a in r2["actions"])
    found_add = any(a == "ADD" for _, a in r2["actions"])
    assert found_update or found_add, r2

    name = store.get_preference("alice", "name")
    assert name == "Alice", name

    results = store.search("what does alice prefer", "alice", k=3)
    assert any(f["kind"] in ("preference", "tool_used") for f in results), results
    print("[OK] mem0_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
