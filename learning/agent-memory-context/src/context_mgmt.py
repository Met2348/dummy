"""Context management: rolling summary / sliding window / prune / RAG-history."""
from __future__ import annotations
from typing import Callable
from common import hash_embed, cosine


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def sliding_window(messages: list[dict], max_turns: int = 20) -> list[dict]:
    if len(messages) <= max_turns:
        return list(messages)
    return list(messages[-max_turns:])


def score_importance(msg: dict) -> int:
    score = 0
    content = msg.get("content", "")
    role = msg.get("role", "user")
    if role == "system":
        score += 100
    if "?" in content:
        score += 3
    if "prefer" in content.lower() or "my name" in content.lower():
        score += 10
    if len(content) > 100:
        score += 1
    return score


def prune(messages: list[dict], target_tokens: int = 2000) -> list[dict]:
    indexed = list(enumerate(messages))
    sorted_msgs = sorted(indexed, key=lambda im: score_importance(im[1]), reverse=True)
    kept: list[tuple[int, dict]] = []
    used = 0
    for idx, msg in sorted_msgs:
        cost = approx_tokens(msg.get("content", ""))
        if used + cost > target_tokens:
            continue
        kept.append((idx, msg))
        used += cost
    kept.sort(key=lambda im: im[0])
    return [m for _, m in kept]


def rolling_summary(
    messages: list[dict],
    summary_fn: Callable[[list[dict]], str],
    keep_recent: int = 5,
    threshold: int = 12,
) -> tuple[str, list[dict]]:
    if len(messages) <= threshold:
        return "", list(messages)
    cut = len(messages) - keep_recent
    old = messages[:cut]
    recent = messages[cut:]
    return summary_fn(old), recent


def rag_history(query: str, history: list[dict], k: int = 5) -> list[dict]:
    if not history:
        return []
    q_vec = hash_embed(query)
    scored = [(m, cosine(q_vec, hash_embed(m.get("content", "")))) for m in history]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    return [m for m, _ in top]


def _mock_summary(messages: list[dict]) -> str:
    bullet = " | ".join(m.get("content", "")[:30] for m in messages[:5])
    return f"[summary of {len(messages)} earlier messages: {bullet}...]"


def _self_test() -> None:
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "My name is Alice and I prefer Anthropic"},
    ] + [{"role": "user", "content": f"chat turn {i}"} for i in range(2, 22)]

    short = sliding_window(msgs, max_turns=5)
    assert len(short) == 5
    assert short[-1]["content"] == "chat turn 21"

    pruned = prune(msgs, target_tokens=200)
    kept_content = " ".join(m["content"] for m in pruned)
    assert "Alice" in kept_content
    assert "agent" in kept_content

    summary, recent = rolling_summary(msgs, _mock_summary, keep_recent=5, threshold=10)
    assert "summary" in summary
    assert len(recent) == 5

    relevant = rag_history("Alice preference", msgs, k=3)
    assert any("Alice" in m["content"] for m in relevant), relevant
    print("[OK] context_mgmt._self_test passed")


if __name__ == "__main__":
    _self_test()
