"""WARC 抽取一致性测试 — 注入 known doc 验证 title/content 抽取."""
from __future__ import annotations

import pytest

pytest.importorskip("trafilatura")

from learning.data_curation_pkg import cc_extract  # type: ignore
# fallback：直接 import 路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cc_extract  # noqa: F811


KNOWN_HTML = b"""<html><body><nav>menu</nav>
<article><p>The mitochondria is the powerhouse of the cell. It produces
adenosine triphosphate through a process called oxidative phosphorylation,
which is the primary energy currency of cellular metabolism.</p></article>
<footer>copyright</footer></body></html>"""


def test_extract_known_doc():
    doc = cc_extract.extract_from_html(KNOWN_HTML, url="http://test/bio")
    assert doc is not None, "trafilatura 抽取返回 None"
    assert "mitochondria" in doc["text"]
    assert "menu" not in doc["text"]
    assert "copyright" not in doc["text"]
    assert doc["url"] == "http://test/bio"


def test_extract_too_short_returns_none():
    """少于 200 字符的应过滤掉."""
    html = b"<html><body><p>too short</p></body></html>"
    assert cc_extract.extract_from_html(html) is None


def test_extract_garbage_returns_none():
    """非 HTML 应安全返回 None."""
    assert cc_extract.extract_from_html(b"random garbage bytes") is None
