"""Capstone smoke test — 1k mock docs 跑通各 stage."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("datasketch")
pytest.importorskip("sentencepiece")

import capstone_mini_corpus as cap     # noqa: E402


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as td:
        report = cap.run_pipeline(out_dir=td, warc_path=None, use_edu=False)
        # 5 stages
        assert len(report) == 5
        names = [r["stage"] for r in report]
        assert names == ["extract", "dedup", "quality", "pii", "tokenize"]
        # 各阶段必须有 n
        for r in report:
            assert "n" in r
        # 至少最后阶段输出 > 0
        # （PII/quality 可能砍掉很多，但 extract 应该有数百）
        assert report[0]["n"] > 100, "extract 应输出 ≥ 100 docs"
        # tokenizer model 应存在
        last = report[-1]
        if last["n"] > 0:
            assert "model" in last
            assert Path(last["model"]).exists()
