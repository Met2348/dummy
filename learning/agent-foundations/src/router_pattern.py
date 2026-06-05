"""Router pattern — LLM picks a sub-handler."""
from __future__ import annotations
from typing import Callable


ROUTER_PROMPT = """Q: {question}
Pick best handler from: {handlers}
Output ONLY the handler name."""


def router(
    question: str,
    llm: Callable[[str], str],
    handlers: dict[str, Callable[[str], str]],
) -> tuple[str, str]:
    """Return (chosen_handler, handler_output)."""
    prompt = ROUTER_PROMPT.format(question=question, handlers=", ".join(handlers))
    pick = llm(prompt).strip().lower()
    pick = pick.split("\n")[0].split()[0] if pick else ""
    chosen = pick if pick in handlers else next(iter(handlers))
    return chosen, handlers[chosen](question)


def embedding_router(
    question: str,
    handlers: dict[str, str],
    handler_fns: dict[str, Callable[[str], str]],
    embed_fn: Callable[[str], list[float]],
) -> tuple[str, str]:
    """Embedding router — pick handler with highest cosine to its description."""
    q_emb = embed_fn(question)
    best_name = ""
    best_score = -1.0
    for name, desc in handlers.items():
        score = _cos(q_emb, embed_fn(desc))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, handler_fns[best_name](question)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def hash_embedding(text: str, dim: int = 16) -> list[float]:
    """Deterministic hash-based mock embedding."""
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = hash(tok)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 + 1e-9
    return [v / norm for v in vec]


def _self_test() -> None:
    from common import make_pattern_llm

    handlers = {
        "math": lambda q: f"MATH: {q}",
        "code": lambda q: f"CODE: {q}",
        "chat": lambda q: f"CHAT: {q}",
    }
    llm = make_pattern_llm([
        (r"2\+3|arithmetic", "math"),
        (r"function|debug", "code"),
        (r".*", "chat"),
    ])

    chosen, out = router("What is 2+3?", llm, handlers)
    assert chosen == "math" and out.startswith("MATH"), (chosen, out)
    chosen, out = router("write a function", llm, handlers)
    assert chosen == "code", (chosen, out)
    chosen, out = router("hello", llm, handlers)
    assert chosen == "chat", (chosen, out)

    desc = {
        "math": "math compute arithmetic calc",
        "code": "python function class debug",
        "chat": "hello hi greet say",
    }
    chosen, _ = embedding_router("compute arithmetic", desc, handlers, hash_embedding)
    assert chosen == "math", chosen
    print("[OK] router_pattern._self_test passed")


if __name__ == "__main__":
    _self_test()
