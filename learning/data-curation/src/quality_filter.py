"""质量过滤 — C4 启发式 + FineWeb-Edu classifier.

教学目标：
    1. C4 + Gopher 10 条启发式规则
    2. FineWeb-Edu classifier 加载 + 打分
    3. 阈值过滤

运行：
    python quality_filter.py --demo
"""
from __future__ import annotations

import argparse
import re
from typing import Sequence


_END_PUNCT = re.compile(r"[\.\?\!\"'\)\]]$")
_BULLET = re.compile(r"^\s*([\-\*•]|\d+[\.\)])\s+")


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 1.0
    return 1 - len(set(grams)) / len(grams)


def heuristic_score(text: str) -> dict:
    """C4 + Gopher 启发式，返回 (passed: bool, reasons: list, stats: dict)."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    words = text.split()
    n_words = len(words)
    n_chars = len(text)

    reasons: list[str] = []

    # 1. 段落 ≥ 5 句（粗略：按 . 切）
    n_sent = text.count(".") + text.count("?") + text.count("!")
    if n_sent < 3:
        reasons.append("too_few_sentences")

    # 2. 末尾有标点
    last = paragraphs[-1] if paragraphs else ""
    if last and not _END_PUNCT.search(last):
        reasons.append("no_end_punct")

    # 3. lorem ipsum
    if "lorem ipsum" in text.lower():
        reasons.append("lorem_ipsum")

    # 4. JS 残留
    if text.count("{") + text.count("}") > 5:
        reasons.append("js_residue")

    # 5. 词数
    if n_words < 50:
        reasons.append("too_few_words")

    # 6. 平均字长 5-15
    avg_wlen = n_chars / max(n_words, 1)
    if not (3 <= avg_wlen <= 15):
        reasons.append(f"bad_avg_wlen={avg_wlen:.1f}")

    # 7. 符号比例
    n_symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    sym_ratio = n_symbols / max(n_chars, 1)
    if sym_ratio > 0.3:
        reasons.append(f"too_many_symbols={sym_ratio:.2f}")

    # 8. bullet 占比
    if paragraphs:
        n_bullet = sum(1 for p in paragraphs if _BULLET.match(p))
        if n_bullet / len(paragraphs) > 0.9:
            reasons.append("mostly_bullets")

    # 9. 重复 5-gram
    ng = repeated_ngram_ratio(text, n=5)
    if ng > 0.2:
        reasons.append(f"high_repeat_5gram={ng:.2f}")

    # 10. 重复 10-char
    ng2 = repeated_ngram_ratio(text, n=10)
    if ng2 > 0.15:
        reasons.append(f"high_repeat_10char={ng2:.2f}")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "n_words": n_words,
        "n_chars": n_chars,
        "avg_wlen": avg_wlen,
        "rep5": ng,
        "sym_ratio": sym_ratio,
    }


# ----- FineWeb-Edu classifier -----
_clf_cache: dict = {}


def fineweb_edu_score(text: str) -> float | None:
    """返回 0-5 教育性 score；模型未装/未下载时返回 None."""
    try:
        if "model" not in _clf_cache:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            _clf_cache["tok"] = AutoTokenizer.from_pretrained(
                "HuggingFaceFW/fineweb-edu-classifier"
            )
            _clf_cache["model"] = AutoModelForSequenceClassification.from_pretrained(
                "HuggingFaceFW/fineweb-edu-classifier"
            )
            _clf_cache["torch"] = torch
        tok, model, torch = _clf_cache["tok"], _clf_cache["model"], _clf_cache["torch"]
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            return float(model(**inputs).logits.item())
    except Exception:
        return None


SAMPLES = [
    "The mitochondria is the powerhouse of the cell. It produces "
    "adenosine triphosphate through oxidative phosphorylation, which fuels "
    "all cellular metabolism. Mitochondria have their own DNA, inherited "
    "maternally, supporting the endosymbiotic theory of their origin.",
    "Click here! Click NOW! Buy this domain. Click here. Click NOW.",
    "lorem ipsum dolor sit amet consectetur adipiscing elit lorem ipsum.",
    "Renewable energy comes from natural sources that are continuously "
    "replenished. Examples include solar, wind, geothermal, hydroelectric, "
    "and biomass. The growth of renewables has accelerated as costs fall.",
]


def run_demo() -> None:
    for i, t in enumerate(SAMPLES):
        h = heuristic_score(t)
        edu = fineweb_edu_score(t)
        print(f"\n--- sample {i} ---")
        print(f"  text: {t[:80]!r}...")
        print(f"  heuristic: passed={h['passed']}  reasons={h['reasons'][:3]}")
        print(f"  fineweb-edu score: {edu if edu is not None else 'SKIP (model not loaded)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
