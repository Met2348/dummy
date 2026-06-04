"""测试 NIAH 用例生成器：长度、深度、可检验性."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from niah_eval import make_niah_query, check_answer, niah_grid


def test_niah_query_contains_needle():
    q, expected = make_niah_query(target_length=500, depth_pct=50,
                                  needle_code="1234")
    assert "1234" in q
    assert check_answer("The code is 1234.", expected) is True


def test_niah_check_answer_negative():
    q, expected = make_niah_query(target_length=200, depth_pct=25,
                                  needle_code="9999")
    assert check_answer("I don't know.", expected) is False


def test_niah_grid_size():
    qs = niah_grid([100, 500], [25, 75], n_samples=3)
    assert len(qs) == 2 * 2 * 3


def test_niah_depth_position():
    for depth in [10, 50, 90]:
        q, expected = make_niah_query(target_length=1000, depth_pct=depth,
                                      needle_code="5555")
        pos = q.find("5555")
        relative = pos / len(q)
        assert 0 < relative < 1
