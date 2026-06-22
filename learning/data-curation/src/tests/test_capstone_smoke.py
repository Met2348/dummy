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
        # extract 应有数百 doc
        assert report[0]["n"] > 100, "extract 应输出 ≥ 100 docs"
        # 流水不能在 dedup/quality 阶段被清空（空语料 no-op 回归保护）：
        # mock 语料须多样且句号结尾，最终 corpus 必须非空且真有 token。
        last = report[-1]
        assert last["n"] > 0, "最终 corpus 为空——mock 数据自相似或缺结尾标点导致流水被清空"
        assert last.get("n_tokens", 0) > 0, "tokenize 阶段 0 token"
        assert "model" in last
        assert Path(last["model"]).exists()
