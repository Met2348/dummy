"""PII detection + redaction.

Two regex backbones + a 'NER' mock that catches names.
Real tools (Presidio, ML NER) do better; this is teaching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# === Regex patterns ===

PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


# Mock "NER" (very crude name detector)
NAMES_PROPER = [
    "Alice", "Bob", "Charlie", "David", "Eve",
    "Frank", "Grace", "Helen", "Ivan", "Judy",
]


def detect_pii(text: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for kind, pat in PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            found[kind] = matches
    # name detection (toy)
    names = [n for n in NAMES_PROPER if re.search(rf"\b{n}\b", text)]
    if names:
        found["person_name"] = names
    return found


def redact(text: str, replace: Dict[str, str] = None) -> str:
    """Replace PII tokens with placeholders."""
    if replace is None:
        replace = {
            "email": "[EMAIL]",
            "phone_us": "[PHONE]",
            "ssn_us": "[SSN]",
            "credit_card": "[CARD]",
            "ipv4": "[IP]",
        }
    out = text
    for kind, pat in PATTERNS.items():
        out = pat.sub(replace.get(kind, "[REDACTED]"), out)
    for n in NAMES_PROPER:
        out = re.sub(rf"\b{n}\b", "[NAME]", out)
    return out


def _self_test() -> int:
    text = "Contact Alice at alice@example.com or 415-555-1234. SSN 123-45-6789. Card 4111-1111-1111-1111. IP 192.168.0.1."
    found = detect_pii(text)
    assert "email" in found and found["email"] == ["alice@example.com"]
    assert "phone_us" in found
    assert "ssn_us" in found
    assert "credit_card" in found
    assert "ipv4" in found
    assert "person_name" in found and "Alice" in found["person_name"]
    red = redact(text)
    assert "alice@example.com" not in red
    assert "[EMAIL]" in red
    assert "[SSN]" in red
    assert "[CARD]" in red
    assert "Alice" not in red
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"pii_redaction.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
