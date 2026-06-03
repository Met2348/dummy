"""Toxicity (Detoxify) + PII (Presidio) 过滤管线.

教学目标：
    1. Detoxify multi-label classifier 接口
    2. Presidio analyzer + anonymizer
    3. 阈值 / replacement 策略

运行：
    python toxicity_pii_filter.py --demo
"""
from __future__ import annotations

import argparse
import re
from typing import Iterable


# ---------- toxicity ----------
_tox_cache: dict = {}


def detoxify_score(text: str) -> dict[str, float] | None:
    try:
        if "model" not in _tox_cache:
            from detoxify import Detoxify
            _tox_cache["model"] = Detoxify("original")
        return _tox_cache["model"].predict(text)
    except Exception:
        return None


def is_toxic(text: str, threshold: float = 0.5) -> tuple[bool, dict | None]:
    s = detoxify_score(text)
    if s is None:
        return False, None
    flagged = (s.get("toxicity", 0) >= threshold or
               s.get("severe_toxicity", 0) >= threshold * 0.5)
    return flagged, s


# ---------- PII ----------
_pii_cache: dict = {}

# 简单 regex 作为 Presidio 不可用时的 fallback
_REGEX_PII = {
    "EMAIL": re.compile(r"\b[\w\.\-_]+@[\w\.\-]+\.[a-zA-Z]{2,}\b"),
    "PHONE_US": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "PHONE_CN": re.compile(r"\b1[3-9]\d{9}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
}


def _regex_anon(text: str) -> tuple[str, list[tuple[str, str]]]:
    found: list[tuple[str, str]] = []
    out = text
    for tag, pat in _REGEX_PII.items():
        for m in pat.finditer(out):
            found.append((tag, m.group(0)))
        out = pat.sub(f"<{tag}>", out)
    return out, found


def anonymize(text: str, use_presidio: bool = True) -> tuple[str, list]:
    if use_presidio:
        try:
            if "analyzer" not in _pii_cache:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                _pii_cache["analyzer"] = AnalyzerEngine()
                _pii_cache["anonymizer"] = AnonymizerEngine()
            results = _pii_cache["analyzer"].analyze(text=text, language="en")
            replaced = _pii_cache["anonymizer"].anonymize(text=text, analyzer_results=results)
            return replaced.text, [(r.entity_type, text[r.start:r.end]) for r in results]
        except Exception:
            pass
    return _regex_anon(text)


def process(text: str, tox_thresh: float = 0.5) -> dict:
    flagged, tox = is_toxic(text, threshold=tox_thresh)
    cleaned, pii = anonymize(text)
    return {
        "drop": flagged,
        "toxicity": tox,
        "text_clean": cleaned,
        "pii_found": pii,
    }


SAMPLES = [
    "Please contact John at john.doe@example.com or call (415) 555-1234.",
    "I hate you idiot, you are the worst.",
    "Renewable energy includes solar wind and hydroelectric sources globally.",
    "SSN 123-45-6789 leaked from the database server at IP 192.168.1.1.",
]


def run_demo() -> None:
    for i, t in enumerate(SAMPLES):
        out = process(t)
        print(f"\n--- sample {i} ---")
        print(f"  in:    {t}")
        print(f"  drop:  {out['drop']}")
        if out["toxicity"]:
            top = max(out["toxicity"].items(), key=lambda x: x[1])
            print(f"  tox_top: {top[0]}={top[1]:.2f}")
        print(f"  pii:   {out['pii_found']}")
        print(f"  out:   {out['text_clean']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
