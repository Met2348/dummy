"""Classifier 测试 — 启发式过滤 / PII 覆盖率."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quality_filter         # noqa: E402
import toxicity_pii_filter    # noqa: E402


GOOD = [
    "The mitochondria is the powerhouse of the cell. It produces "
    "adenosine triphosphate through oxidative phosphorylation. This "
    "process fuels all cellular metabolism in animals. Many scientists "
    "have studied this fundamental biological mechanism in depth daily.",
    "Renewable energy comes from naturally replenished sources. Solar, "
    "wind, hydro, and geothermal are the main pillars. Costs of solar "
    "panels have dropped by 90% over the past decade. The energy mix "
    "will continue to evolve rapidly in coming years globally.",
]

BAD = [
    "lorem ipsum dolor sit amet consectetur adipiscing elit lorem ipsum dolor",
    "click here CLICK HERE Click Here CLICK click click here NOW NOW NOW",
    "a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a",
]


def test_heuristic_passes_good():
    for t in GOOD:
        h = quality_filter.heuristic_score(t)
        assert h["passed"], f"good text should pass, reasons: {h['reasons']}"


def test_heuristic_fails_bad():
    for t in BAD:
        h = quality_filter.heuristic_score(t)
        assert not h["passed"], f"bad text should fail, got passed: {t!r}"


def test_pii_email_replaced():
    text = "Contact me at jane.smith@example.com for details."
    cleaned, found = toxicity_pii_filter.anonymize(text)
    # regex fallback 保证 email 被识别
    assert "@example.com" not in cleaned or "<EMAIL" in cleaned, cleaned
    assert any("EMAIL" in tag.upper() for tag, _ in found), f"email not found: {found}"


def test_pii_phone_replaced():
    text = "Call us at (415) 555-1234 anytime today."
    cleaned, found = toxicity_pii_filter.anonymize(text)
    # 至少有 1 个 phone 被识别（presidio 或 regex 路径之一）
    has_phone = any("PHONE" in tag.upper() for tag, _ in found)
    if not has_phone:
        # presidio 路径下可能识别为 US 风格
        has_phone = "<PHONE" in cleaned or "<US_PHONE" in cleaned
    assert has_phone, f"phone not detected: {found}, cleaned={cleaned}"
