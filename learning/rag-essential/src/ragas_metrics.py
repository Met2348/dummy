"""RAGAS-style metrics (mock implementations)."""
from __future__ import annotations
from common import tokenize, hash_embed, cosine, RetrievalResult
import re


def split_claims(text: str) -> list[str]:
    sents = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [s for s in sents if s.strip()]


def faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer.strip():
        return 0.0
    claims = split_claims(answer)
    if not claims:
        return 0.0
    ctx_blob = " ".join(contexts).lower()
    supported = 0
    for claim in claims:
        c_tokens = [t for t in tokenize(claim) if len(t) > 2]
        if not c_tokens:
            continue
        matched = sum(1 for t in c_tokens if t in ctx_blob)
        if matched / len(c_tokens) >= 0.5:
            supported += 1
    return supported / len(claims)


def answer_relevancy(answer: str, query: str) -> float:
    return cosine(hash_embed(answer), hash_embed(query))


def context_precision(query: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    q_tokens = set(t for t in tokenize(query) if len(t) > 2)
    if not q_tokens:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for i, ctx in enumerate(contexts):
        c_tokens = set(tokenize(ctx))
        rel = len(q_tokens & c_tokens) / len(q_tokens)
        weight = 1.0 / (i + 1)
        weighted_sum += rel * weight
        weight_total += weight
    return weighted_sum / weight_total


def context_recall(gt_answer: str, contexts: list[str]) -> float:
    if not gt_answer.strip():
        return 0.0
    gt_claims = split_claims(gt_answer)
    if not gt_claims:
        return 0.0
    ctx_blob = " ".join(contexts).lower()
    recallable = 0
    for claim in gt_claims:
        c_tokens = [t for t in tokenize(claim) if len(t) > 2]
        if not c_tokens:
            continue
        matched = sum(1 for t in c_tokens if t in ctx_blob)
        if matched / len(c_tokens) >= 0.5:
            recallable += 1
    return recallable / len(gt_claims)


def ragas_eval(query: str, answer: str, contexts: list[str], gt_answer: str = "") -> dict:
    out = {
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevancy": answer_relevancy(answer, query),
        "context_precision": context_precision(query, contexts),
    }
    if gt_answer:
        out["context_recall"] = context_recall(gt_answer, contexts)
    out["mean"] = sum(out.values()) / len(out)
    return out


def _self_test() -> None:
    contexts = [
        "Anthropic was founded in 2021 by Dario and Daniela Amodei.",
        "Claude is an AI assistant made by Anthropic.",
    ]
    answer = "Anthropic was founded in 2021. Claude is by Anthropic."
    query = "Who founded Anthropic?"
    gt = "Anthropic was founded in 2021 by Dario Amodei."

    out = ragas_eval(query, answer, contexts, gt)
    assert out["faithfulness"] > 0.5, out
    assert out["answer_relevancy"] > 0.3, out
    assert out["context_precision"] > 0.3, out
    assert out["context_recall"] > 0.5, out
    assert out["mean"] > 0.4, out

    out_bad = ragas_eval(query, "I have no idea what that is.", [], gt)
    assert out_bad["faithfulness"] < 0.3, out_bad
    print("[OK] ragas_metrics._self_test passed")


if __name__ == "__main__":
    _self_test()
